"""Train/test split utilities for RF fingerprinting.

The standard sklearn `train_test_split` with `random_state` is unsafe for
SDR datasets. The DeepResearch report (2026-04-27) explicitly flagged
ORACLE-style data leakage: random splits over time-adjacent slices from
the same recording cause models to memorize burst-internal correlations.
Published 99% accuracies are inflated by 10-30 pp once leakage is fixed.

This module provides safe alternatives:

    grouped_split(X, y, groups, ...)
        Split on a grouping key — same group never appears in both
        train and test. Use `groups='burst_id'`, 'recording_id',
        'session_id', etc.

    day_aware_split(X, y, days, train_days, test_days)
        Train on Day-N, test on Day-M. The strongest cross-day
        generalization test, per WiSig / OSU NetSTAR / ANDRO findings.

    receiver_held_out_split(X, y, receivers, train_rxs, test_rxs)
        Train on subset of receivers, test on disjoint subset. Tests
        receiver-agnostic transmitter ID — exposes Rx fingerprint
        contamination.

    leakage_audit(X_train, X_test, indices_train, indices_test, groups)
        Programmatically check that no group appears in both splits.

All splitters return integer index arrays (not data slices) so they
compose with sklearn's cross_val_score, GridSearchCV, etc.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Iterable

import numpy as np


# ---- Grouped split -----------------------------------------------------


def grouped_split(
    n_samples: int,
    groups: Sequence,
    test_size: float = 0.3,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split such that no group_id appears in both train and test.

    Parameters
    ----------
    n_samples : total number of samples
    groups : sequence of length n_samples — group_id per sample
    test_size : fraction of UNIQUE GROUPS to put in test (not samples)
    random_state : seed for group shuffling

    Returns
    -------
    train_idx, test_idx : np.ndarray of integer indices

    Example
    -------
    For ORACLE-style data: each USRP burst is a group_id. Random sample
    splitting leaks; group splitting is safe.
    """
    groups = np.asarray(groups)
    if len(groups) != n_samples:
        raise ValueError(f"groups length {len(groups)} != n_samples {n_samples}")
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    rng = np.random.default_rng(random_state)
    unique = np.unique(groups)
    rng.shuffle(unique)
    n_test_groups = max(1, int(round(test_size * len(unique))))
    test_groups = set(unique[:n_test_groups])

    test_mask = np.isin(groups, list(test_groups))
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    if len(train_idx) == 0:
        raise ValueError("No samples assigned to train set; check group structure.")
    if len(test_idx) == 0:
        raise ValueError("No samples assigned to test set; check group structure.")
    return train_idx, test_idx


def grouped_kfold_indices(
    n_samples: int,
    groups: Sequence,
    n_splits: int = 5,
    random_state: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate K-fold splits where groups are kept in single folds.

    Returns list of (train_idx, test_idx) tuples. Useful for grouped
    cross-validation in transmitter-fingerprinting evaluation.
    """
    groups = np.asarray(groups)
    rng = np.random.default_rng(random_state)
    unique = np.unique(groups)
    rng.shuffle(unique)
    fold_size = len(unique) // n_splits
    if fold_size == 0:
        raise ValueError(f"Too few unique groups ({len(unique)}) for {n_splits} folds")
    folds = []
    for k in range(n_splits):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_splits - 1 else len(unique)
        test_groups = set(unique[start:end])
        test_mask = np.isin(groups, list(test_groups))
        folds.append((np.where(~test_mask)[0], np.where(test_mask)[0]))
    return folds


# ---- Day-aware split --------------------------------------------------


def day_aware_split(
    n_samples: int,
    days: Sequence,
    train_days: Sequence | None = None,
    test_days: Sequence | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train on specified days, test on disjoint days.

    Per WiSig / OSU NetSTAR findings: Day-1-trained / Day-2-tested
    accuracy drops 30-50 pp on most datasets. Reporting cross-day
    generalization is essential for honest fingerprinting claims.

    Parameters
    ----------
    days : sequence of day labels (e.g. dates as strings, or integers)
    train_days, test_days : explicit day sets. If None: split on first
        half of unique days for train, rest for test.
    """
    days = np.asarray(days)
    if len(days) != n_samples:
        raise ValueError(f"days length {len(days)} != n_samples {n_samples}")

    if train_days is None and test_days is None:
        unique_days = sorted(np.unique(days), key=str)
        n_train = max(1, len(unique_days) // 2)
        train_days = unique_days[:n_train]
        test_days = unique_days[n_train:]
    elif train_days is None or test_days is None:
        raise ValueError("Must provide both train_days and test_days, or neither.")

    train_set = set(train_days)
    test_set = set(test_days)
    overlap = train_set & test_set
    if overlap:
        raise ValueError(f"train_days and test_days overlap: {overlap}")

    train_mask = np.isin(days, list(train_set))
    test_mask = np.isin(days, list(test_set))

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    if len(train_idx) == 0:
        raise ValueError(f"No samples for train_days={train_days}")
    if len(test_idx) == 0:
        raise ValueError(f"No samples for test_days={test_days}")

    return train_idx, test_idx


# ---- Receiver-held-out split ------------------------------------------


def receiver_held_out_split(
    n_samples: int,
    receivers: Sequence,
    train_rxs: Sequence | None = None,
    test_rxs: Sequence | None = None,
    test_size: float = 0.5,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split so the test receivers are unseen at training time.

    Tests receiver-agnostic transmitter ID — exposes whether the model
    is learning Tx fingerprints (good) or Rx artifacts (bad). WiSig
    explicitly demonstrates that single-receiver training generalizes
    poorly across receivers.
    """
    receivers = np.asarray(receivers)
    if len(receivers) != n_samples:
        raise ValueError(f"receivers length {len(receivers)} != n_samples {n_samples}")

    if train_rxs is None and test_rxs is None:
        rng = np.random.default_rng(random_state)
        unique = np.unique(receivers)
        rng.shuffle(unique)
        n_test = max(1, int(round(test_size * len(unique))))
        test_rxs = unique[:n_test]
        train_rxs = unique[n_test:]
        if len(train_rxs) == 0:
            raise ValueError(f"Need at least 2 receivers, got {len(unique)}")

    train_set = set(train_rxs)
    test_set = set(test_rxs)
    if train_set & test_set:
        raise ValueError(f"train_rxs and test_rxs overlap: {train_set & test_set}")

    train_mask = np.isin(receivers, list(train_set))
    test_mask = np.isin(receivers, list(test_set))
    return np.where(train_mask)[0], np.where(test_mask)[0]


# ---- Leakage audit ----------------------------------------------------


def leakage_audit(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: Sequence | None = None,
    days: Sequence | None = None,
    receivers: Sequence | None = None,
) -> dict:
    """Programmatically check that no group/day/receiver appears in both splits.

    Returns dict of leakage stats per axis. Use as a final sanity check
    before fitting any classifier.

    Example
    -------
        idx_tr, idx_te = grouped_split(...)
        report = leakage_audit(idx_tr, idx_te, groups=burst_ids,
                               receivers=rx_ids)
        assert report['groups']['leakage'] == 0
    """
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    if np.intersect1d(train_idx, test_idx).size > 0:
        warnings.warn("train_idx and test_idx share sample indices — direct "
                      "leakage detected!")

    out = {
        "n_train_samples": int(len(train_idx)),
        "n_test_samples": int(len(test_idx)),
        "sample_overlap": int(len(np.intersect1d(train_idx, test_idx))),
    }

    for name, axis in (("groups", groups), ("days", days),
                       ("receivers", receivers)):
        if axis is None:
            continue
        axis = np.asarray(axis)
        train_axis = set(axis[train_idx].tolist())
        test_axis = set(axis[test_idx].tolist())
        overlap = train_axis & test_axis
        out[name] = {
            "n_train_unique": len(train_axis),
            "n_test_unique": len(test_axis),
            "leakage": len(overlap),
            "overlapping": sorted(overlap)[:10] if overlap else [],
        }

    return out


__all__ = [
    "grouped_split",
    "grouped_kfold_indices",
    "day_aware_split",
    "receiver_held_out_split",
    "leakage_audit",
]
