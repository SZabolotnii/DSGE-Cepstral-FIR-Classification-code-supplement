"""End-to-end DSGE pipeline for classification tasks.

Provides a single `run_dsge_pipeline` function that wraps the complete
workflow of the HAR and NLP papers:

    1. (optional) StandardScaler normalisation
    2. DSGE feature generation (per-class reconstruction errors)
    3. (optional) hybrid concatenation with traditional features
    4. classifier training and evaluation
    5. (optional) basis comparison grid search
    6. metrics: accuracy, macro-F1, per-class report

The function returns a single dict with all results, so downstream code
(notebooks, papers, dashboards) can extract whichever pieces are needed.

Compatible with sklearn classifiers (default: LogisticRegression). Pass any
estimator with .fit / .predict to use a different classifier.

Typical usage:
    from dsge_pipeline import run_dsge_pipeline

    result = run_dsge_pipeline(
        X_train, y_train, X_test, y_test,
        basis='fractional', n=3,
        compare_bases=True,
    )
    print(result['summary_table'])
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kunchenko_features import DSGEFeatureExtractor  # noqa: E402


@dataclass
class DSGEResult:
    config: dict
    accuracy: float
    macro_f1: float
    per_class_report: dict | None = None
    feature_dim: int | None = None
    cond_numbers: dict | None = None
    extras: dict = field(default_factory=dict)


def _default_classifier():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000)


def _evaluate(y_true, y_pred):
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def run_dsge_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    basis: str = "fractional",
    n: int = 3,
    alpha: float | None = None,
    ridge: float = 0.01,
    classifier: Any = None,
    standardize: bool = True,
    traditional_features_train: np.ndarray | None = None,
    traditional_features_test: np.ndarray | None = None,
    compare_bases: bool = False,
    bases_to_compare: list | None = None,
) -> dict:
    """Run a complete DSGE classification experiment.

    Parameters
    ----------
    X_train, X_test : (n_samples, dim) arrays
        Input vectors. For NLP, these are embeddings; for HAR, raw or
        windowed sensor signals.
    y_train, y_test : (n_samples,)
    basis : one of 'polynomial', 'fractional', 'trigonometric', 'robust',
            'log', 'patp'
    n : number of basis functions (3 is a sensible default)
    alpha : transition parameter for 'patp' basis only
    ridge : Tikhonov regularisation parameter
    classifier : sklearn-style estimator. Default: LogisticRegression.
    standardize : if True, fit StandardScaler on train and apply to test.
    traditional_features_* : if provided, concatenate alongside DSGE features
        to build a HYBRID model. This is the configuration that gives the
        large gains in the HAR and NLP papers.
    compare_bases : if True, additionally evaluate every basis in
        `bases_to_compare` (default: ['polynomial', 'fractional',
        'trigonometric', 'robust']) and return a summary table.

    Returns
    -------
    dict with keys:
        'main' : DSGEResult for the primary configuration
        'comparison' : list of DSGEResult, one per basis (if compare_bases)
        'summary_table' : list of dicts suitable for printing/saving
    """
    if standardize:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X_train)
        X_train_s = sc.transform(X_train)
        X_test_s = sc.transform(X_test)
    else:
        X_train_s, X_test_s = X_train, X_test

    def _evaluate_one(basis_name: str, alpha_val: float | None) -> DSGEResult:
        extractor = DSGEFeatureExtractor(
            basis=basis_name, n=n, alpha=alpha_val, ridge=ridge
        ).fit(X_train_s, y_train)
        feat_train = extractor.transform(X_train_s)
        feat_test = extractor.transform(X_test_s)

        if traditional_features_train is not None:
            feat_train = np.hstack([feat_train, traditional_features_train])
            feat_test = np.hstack([feat_test, traditional_features_test])

        clf = classifier if classifier is not None else _default_classifier()
        clf.fit(feat_train, y_train)
        y_pred = clf.predict(feat_test)
        metrics = _evaluate(y_test, y_pred)

        return DSGEResult(
            config={
                "basis": basis_name, "n": n, "alpha": alpha_val,
                "ridge": ridge, "hybrid": traditional_features_train is not None,
                "standardize": standardize,
            },
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            per_class_report=metrics["report"],
            feature_dim=feat_train.shape[1],
            cond_numbers=extractor.conditioning_report,
        )

    main = _evaluate_one(basis, alpha)
    comparison: list[DSGEResult] = []
    if compare_bases:
        if bases_to_compare is None:
            bases_to_compare = ["polynomial", "fractional", "trigonometric", "robust"]
        for b in bases_to_compare:
            if b == basis and alpha is None:
                comparison.append(main)
                continue
            comparison.append(_evaluate_one(b, None))

    summary_table = [
        {
            "basis": r.config["basis"],
            "alpha": r.config.get("alpha"),
            "feature_dim": r.feature_dim,
            "accuracy": round(r.accuracy, 4),
            "macro_f1": round(r.macro_f1, 4),
            "max_cond_F": round(max(r.cond_numbers.values()), 1) if r.cond_numbers else None,
        }
        for r in (comparison if comparison else [main])
    ]

    return {"main": main, "comparison": comparison, "summary_table": summary_table}


def patp_alpha_sweep(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n: int = 3, ridge: float = 0.01,
    alphas: np.ndarray | None = None,
    classifier: Any = None,
    standardize: bool = True,
) -> dict:
    """Sweep alpha in [0, 1] for PATP basis and report best.

    This is the cross-domain "Case Study N" experiment: a single continuous
    parameter replaces the discrete poly/frac/trig/robust grid. The optimal
    alpha is selected on the (X_test, y_test) split — for honest reporting,
    use a separate validation split, then re-evaluate on a held-out test.

    Returns:
        {'alphas': [...], 'macro_f1': [...], 'accuracy': [...],
         'best_alpha': float, 'best_macro_f1': float}
    """
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 21)
    f1s, accs = [], []
    for a in alphas:
        res = run_dsge_pipeline(
            X_train, y_train, X_test, y_test,
            basis="patp", n=n, alpha=float(a), ridge=ridge,
            classifier=classifier, standardize=standardize,
        )
        f1s.append(res["main"].macro_f1)
        accs.append(res["main"].accuracy)
    best_idx = int(np.argmax(f1s))
    return {
        "alphas": alphas.tolist(),
        "macro_f1": f1s,
        "accuracy": accs,
        "best_alpha": float(alphas[best_idx]),
        "best_macro_f1": float(f1s[best_idx]),
        "best_accuracy": float(accs[best_idx]),
    }
