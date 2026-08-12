"""Training loop for TabularTransformer.

The target dtype and shape are chosen from the task, not assumed:

  regression      float32, shape (B, 1), MSELoss
  classification  int64,   shape (B,),   CrossEntropyLoss

Getting this wrong is the single most common way this model fails to train --
CrossEntropyLoss will not accept a float (B, 1) target.

Regression targets are standardised internally (mean/std taken from the
training split only) and predictions are mapped back to the original units.
Without this, a target such as glucose in mg/dL leaves the network fighting to
learn a ~140 unit bias through MSE, and it will sit near zero for many epochs
and report a nonsense RMSE.

Other things this loop does that matter: batches are shuffled every epoch,
the validation split is seeded from the caller's seed rather than a hardcoded
constant, class weights are derived from the training split only, and the best
checkpoint is deep-copied (a live ``state_dict()`` reference keeps mutating).
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class TransformerTrainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 10,
        val_split: float = 0.2,
        device: Optional[str] = None,
        class_weight: Optional[np.ndarray] = None,
        grad_clip: float = 1.0,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.model = model
        self.regression = bool(getattr(model, "regression", True))
        self.batch_size = max(2, int(batch_size))
        self.epochs = max(1, int(epochs))
        self.patience = max(1, int(patience))
        self.val_split = float(val_split)
        self.grad_clip = float(grad_clip)
        self.seed = int(seed)
        self.verbose = verbose
        self.class_weight = class_weight

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=max(2, self.patience // 3)
        )
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}
        # Regression target standardisation, fitted on the training split only.
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.best_val_loss = float("inf")
        self.best_state: Optional[Dict] = None
        self.best_epoch: Optional[int] = None
        self.epochs_run = 0

    # ------------------------------------------------------------------
    def _targets(self, y: np.ndarray) -> torch.Tensor:
        if self.regression:
            scaled = (np.asarray(y, dtype=np.float64) - self.y_mean_) / self.y_std_
            return torch.as_tensor(scaled.astype(np.float32)).view(-1, 1)
        return torch.as_tensor(np.asarray(y).astype(np.int64)).view(-1)

    def _criterion(self) -> nn.Module:
        if self.regression:
            return nn.MSELoss()
        weight = None
        if self.class_weight is not None:
            weight = torch.as_tensor(
                np.asarray(self.class_weight, dtype=np.float32), device=self.device
            )
        return nn.CrossEntropyLoss(weight=weight)

    def _epoch(self, X, y, criterion, train: bool) -> float:
        self.model.train(train)
        n = len(X)
        order = torch.randperm(n) if train else torch.arange(n)
        total, n_batches = 0.0, 0
        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            if train and len(idx) < 2:
                continue  # a 1-row batch breaks normalisation layers
            xb = X[idx].to(self.device)
            yb = y[idx].to(self.device)
            with torch.set_grad_enabled(train):
                out = self.model(xb)
                loss = criterion(out, yb)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
            total += float(loss.item())
            n_batches += 1
        return total / max(1, n_batches)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> Dict:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if X_val is None and self.val_split > 0 and len(X) >= 10:
            stratify = y if not self.regression and len(np.unique(y)) > 1 else None
            try:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X, y, test_size=self.val_split,
                    random_state=self.seed, stratify=stratify,
                )
            except ValueError:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X, y, test_size=self.val_split, random_state=self.seed
                )
        elif X_val is None:
            X_tr, y_tr, X_val, y_val = X, y, X, y
            logger.warning(
                "No validation split (n=%d); early stopping will watch training loss.", len(X)
            )
        else:
            X_tr, y_tr = X, y
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val)

        if not self.regression and self.class_weight is None:
            classes, counts = np.unique(y_tr, return_counts=True)
            if len(classes) > 1 and counts.min() / counts.max() < 0.4:
                self.class_weight = (len(y_tr) / (len(classes) * counts)).astype(np.float32)
                logger.info("Imbalanced classes; using weights %s", np.round(self.class_weight, 3))

        if self.regression:
            self.y_mean_ = float(np.mean(np.asarray(y_tr, dtype=np.float64)))
            std = float(np.std(np.asarray(y_tr, dtype=np.float64)))
            self.y_std_ = std if std > 1e-9 else 1.0
            logger.info(
                "Standardising regression target (train mean %.4g, std %.4g); "
                "predictions are returned in the original units.",
                self.y_mean_, self.y_std_,
            )

        Xt = torch.as_tensor(X_tr)
        yt = self._targets(y_tr)
        Xv = torch.as_tensor(np.asarray(X_val, dtype=np.float32))
        yv = self._targets(y_val)
        criterion = self._criterion()

        torch.manual_seed(self.seed)
        bad_epochs = 0
        for epoch in range(self.epochs):
            train_loss = self._epoch(Xt, yt, criterion, train=True)
            val_loss = self._epoch(Xv, yv, criterion, train=False)
            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(float(self.optimizer.param_groups[0]["lr"]))
            self.epochs_run = epoch + 1

            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                logger.info(
                    "  epoch %3d | train %.5f | val %.5f", epoch + 1, train_loss, val_loss
                )

            if val_loss < self.best_val_loss - 1e-6:
                self.best_val_loss = val_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch + 1
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    logger.info("Early stopping at epoch %d.", epoch + 1)
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            logger.info(
                "Restored best checkpoint from epoch %s (val loss %.5f).",
                self.best_epoch, self.best_val_loss,
            )
        return self.history

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Regression -> (n,) values. Classification -> (n, n_classes) probabilities."""
        self.model.eval()
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        outputs = []
        for start in range(0, len(X_t), self.batch_size):
            out = self.model(X_t[start : start + self.batch_size].to(self.device))
            outputs.append(out.cpu())
        logits = torch.cat(outputs) if outputs else torch.empty(0)
        if self.regression:
            return logits.numpy().ravel() * self.y_std_ + self.y_mean_
        return torch.softmax(logits, dim=1).numpy()

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        if self.regression:
            raise RuntimeError("predict_labels() is classification-only.")
        return self.predict(X).argmax(axis=1)
