"""Train/test and cross-validation splitting.

Three regimes, chosen explicitly rather than guessed:

``time_series``  chronological cut, optional gap, expanding-window inner CV.
``group``        no subject appears in both train and test.
``stratified``   class-balanced random split (classification only).
``random``       plain random split (regression without groups).

The stratified path is guarded: stratifying on a continuous target raises in
scikit-learn, so we check the task first instead of letting it crash.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

logger = logging.getLogger(__name__)

VALID_CV_TYPES = ("time_series", "group", "stratified", "random")


@dataclass
class SplitResult:
    """Outer train/test indices plus inner CV folds defined on the train block.

    ``cv_splits`` indices are positions *within the training block*, not within
    the original array. ``train_idx[cv_splits[0][0]]`` gives original positions.
    """

    train_idx: np.ndarray
    test_idx: np.ndarray
    cv_splits: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    cv_type: str = "random"
    notes: List[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "cv_type": self.cv_type,
            "n_train": int(len(self.train_idx)),
            "n_test": int(len(self.test_idx)),
            "n_cv_folds": len(self.cv_splits),
            "notes": list(self.notes),
        }


def resolve_cv_type(
    requested: str,
    is_classification: bool,
    has_groups: bool,
    has_time: bool,
) -> Tuple[str, List[str]]:
    """Pick a workable cv_type, explaining any downgrade."""
    notes: List[str] = []
    cv_type = (requested or "random").lower()
    if cv_type not in VALID_CV_TYPES:
        notes.append(f"Unknown cv_type '{requested}'; falling back to 'random'.")
        cv_type = "random"
    if cv_type == "time_series" and not has_time:
        notes.append(
            "cv_type='time_series' requested but no ordering key was supplied; "
            "rows are assumed to already be in chronological order."
        )
    if cv_type == "group" and not has_groups:
        notes.append("cv_type='group' requested but no groups given; using 'random'.")
        cv_type = "random"
    if cv_type == "stratified" and not is_classification:
        notes.append("cv_type='stratified' is classification-only; using 'random'.")
        cv_type = "random"
    return cv_type, notes


def make_splits(
    n_samples: int,
    y: Optional[np.ndarray] = None,
    groups: Optional[Sequence] = None,
    cv_type: str = "random",
    test_size: float = 0.2,
    n_splits: int = 5,
    gap: int = 0,
    random_state: int = 42,
    is_classification: bool = True,
    order: Optional[np.ndarray] = None,
) -> SplitResult:
    """Build the outer split and inner CV folds.

    ``order`` is an optional sortable key (e.g. window end timestamps). When
    given with ``cv_type='time_series'`` the data is sorted by it first, so the
    chronological cut is genuinely chronological rather than dependent on row
    order in the source file.
    """
    if n_samples < 5:
        raise ValueError(f"Need at least 5 samples to split; got {n_samples}.")

    groups_arr = np.asarray(groups) if groups is not None else None
    has_groups = groups_arr is not None and len(np.unique(groups_arr)) > 1
    cv_type, notes = resolve_cv_type(
        cv_type, is_classification, has_groups, order is not None
    )

    index = np.arange(n_samples)
    if cv_type == "time_series" and order is not None:
        index = index[np.argsort(np.asarray(order), kind="stable")]
        notes.append("Rows sorted by the supplied ordering key before splitting.")

    n_splits = max(2, min(int(n_splits), n_samples // 2))

    # ---------------- outer split ----------------
    if cv_type == "time_series":
        n_test = max(1, int(round(n_samples * test_size)))
        n_train = n_samples - n_test - int(gap)
        if n_train < 2:
            raise ValueError(
                f"time_series split leaves {n_train} training rows "
                f"(n={n_samples}, test_size={test_size}, gap={gap})."
            )
        train_idx = index[:n_train]
        test_idx = index[n_train + int(gap):]
    elif cv_type == "group":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr, te = next(gss.split(index, y, groups_arr))
        train_idx, test_idx = index[tr], index[te]
    elif cv_type == "stratified":
        train_idx, test_idx = train_test_split(
            index, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        train_idx, test_idx = train_test_split(
            index, test_size=test_size, random_state=random_state, shuffle=True
        )

    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)

    # ---------------- inner CV on the training block ----------------
    cv_splits: List[Tuple[np.ndarray, np.ndarray]] = []
    y_train = np.asarray(y)[train_idx] if y is not None else None
    g_train = groups_arr[train_idx] if groups_arr is not None else None

    try:
        if cv_type == "time_series":
            splitter = TimeSeriesSplit(n_splits=n_splits, gap=int(gap))
            cv_splits = [(a, b) for a, b in splitter.split(train_idx)]
        elif cv_type == "group":
            n_groups = len(np.unique(g_train))
            k = max(2, min(n_splits, n_groups))
            splitter = GroupKFold(n_splits=k)
            cv_splits = [(a, b) for a, b in splitter.split(train_idx, y_train, g_train)]
        elif cv_type == "stratified":
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            cv_splits = [(a, b) for a, b in splitter.split(train_idx, y_train)]
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            cv_splits = [(a, b) for a, b in splitter.split(train_idx)]
    except ValueError as exc:
        notes.append(f"Inner CV construction failed ({exc}); falling back to KFold.")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_splits = [(a, b) for a, b in splitter.split(train_idx)]

    result = SplitResult(train_idx, test_idx, cv_splits, cv_type, notes)
    for note in notes:
        logger.warning("Split: %s", note)
    logger.info(
        "Split (%s): %d train / %d test, %d inner folds",
        cv_type, len(train_idx), len(test_idx), len(cv_splits),
    )
    return result


def check_group_disjoint(
    train_idx: np.ndarray, test_idx: np.ndarray, groups: Optional[Sequence]
) -> Optional[List]:
    """Return groups appearing on both sides of the split, or None."""
    if groups is None:
        return None
    groups_arr = np.asarray(groups)
    overlap = set(groups_arr[train_idx].tolist()) & set(groups_arr[test_idx].tolist())
    return sorted(overlap) if overlap else None
