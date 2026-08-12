"""Training loop for MultimodalModel over a DataLoader of dict batches.

Mirrors TransformerTrainer: task-correct target dtypes, regression target
standardisation, seeded shuffling, best-checkpoint restore.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


class MultimodalTrainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 16,
        epochs: int = 30,
        patience: int = 8,
        device: Optional[str] = None,
        class_weight: Optional[np.ndarray] = None,
        grad_clip: float = 1.0,
        seed: int = 42,
        num_workers: int = 0,
        verbose: bool = True,
    ):
        self.model = model
        self.regression = bool(getattr(model, "regression", True))
        self.batch_size = max(2, int(batch_size))
        self.epochs = max(1, int(epochs))
        self.patience = max(1, int(patience))
        self.grad_clip = grad_clip
        self.seed = int(seed)
        self.num_workers = num_workers
        self.verbose = verbose
        self.class_weight = class_weight

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=learning_rate,
                                           weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=max(2, self.patience // 3)
        )
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.best_val_loss = float("inf")
        self.best_state: Optional[Dict] = None
        self.best_epoch: Optional[int] = None
        self.epochs_run = 0
        self.y_mean_, self.y_std_ = 0.0, 1.0

    # ------------------------------------------------------------------
    def _to_device(self, inputs: Dict) -> Dict:
        return {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

    def _criterion(self) -> nn.Module:
        if self.regression:
            return nn.MSELoss()
        weight = (torch.as_tensor(np.asarray(self.class_weight, dtype=np.float32),
                                  device=self.device)
                  if self.class_weight is not None else None)
        return nn.CrossEntropyLoss(weight=weight)

    def _prepare_target(self, y: torch.Tensor) -> torch.Tensor:
        if self.regression:
            scaled = (y.double() - self.y_mean_) / self.y_std_
            return scaled.float().view(-1, 1)
        return y.long().view(-1)

    def _run_epoch(self, loader: DataLoader, criterion, train: bool) -> float:
        self.model.train(train)
        total, batches = 0.0, 0
        for inputs, presence, targets in loader:
            if train and targets.shape[0] < 2:
                continue
            inputs = self._to_device(inputs)
            presence = {k: v.to(self.device) for k, v in presence.items()}
            targets = self._prepare_target(targets).to(self.device)
            with torch.set_grad_enabled(train):
                outputs = self.model(inputs, presence)
                loss = criterion(outputs, targets)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.grad_clip,
                        )
                    self.optimizer.step()
            total += float(loss.item())
            batches += 1
        return total / max(1, batches)

    # ------------------------------------------------------------------
    def fit(self, dataset, train_indices, val_indices) -> Dict:
        from src.data.multimodal import collate

        y_train = np.asarray([dataset.y[i] for i in train_indices], dtype=np.float64)
        if self.regression:
            self.y_mean_ = float(y_train.mean())
            std = float(y_train.std())
            self.y_std_ = std if std > 1e-9 else 1.0
            logger.info("Standardising target (mean %.4g, std %.4g).",
                        self.y_mean_, self.y_std_)
        elif self.class_weight is None:
            classes, counts = np.unique(y_train.astype(int), return_counts=True)
            if len(classes) > 1 and counts.min() / counts.max() < 0.4:
                self.class_weight = (len(y_train) / (len(classes) * counts)).astype(np.float32)
                logger.info("Imbalanced classes; weights %s", np.round(self.class_weight, 3))

        generator = torch.Generator().manual_seed(self.seed)
        train_loader = DataLoader(
            Subset(dataset, list(train_indices)), batch_size=self.batch_size,
            shuffle=True, collate_fn=collate, num_workers=self.num_workers,
            generator=generator,
        )
        val_loader = DataLoader(
            Subset(dataset, list(val_indices)), batch_size=self.batch_size,
            shuffle=False, collate_fn=collate, num_workers=self.num_workers,
        )
        criterion = self._criterion()
        torch.manual_seed(self.seed)
        bad_epochs = 0

        for epoch in range(self.epochs):
            train_loss = self._run_epoch(train_loader, criterion, True)
            val_loss = self._run_epoch(val_loader, criterion, False)
            self.scheduler.step(val_loss)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.epochs_run = epoch + 1

            if self.verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                logger.info("  epoch %3d | train %.5f | val %.5f",
                            epoch + 1, train_loss, val_loss)

            if val_loss < self.best_val_loss - 1e-6:
                self.best_val_loss, self.best_epoch = val_loss, epoch + 1
                self.best_state = copy.deepcopy(self.model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    logger.info("Early stopping at epoch %d.", epoch + 1)
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            logger.info("Restored best checkpoint (epoch %s, val loss %.5f).",
                        self.best_epoch, self.best_val_loss)
        return self.history

    @torch.no_grad()
    def predict(self, dataset, indices) -> np.ndarray:
        from src.data.multimodal import collate

        self.model.eval()
        loader = DataLoader(
            Subset(dataset, list(indices)), batch_size=self.batch_size,
            shuffle=False, collate_fn=collate, num_workers=self.num_workers,
        )
        outputs = []
        for inputs, presence, _ in loader:
            inputs = self._to_device(inputs)
            presence = {k: v.to(self.device) for k, v in presence.items()}
            outputs.append(self.model(inputs, presence).cpu())
        if not outputs:
            return np.empty(0)
        logits = torch.cat(outputs)
        if self.regression:
            return logits.numpy().ravel() * self.y_std_ + self.y_mean_
        return torch.softmax(logits, dim=1).numpy()
