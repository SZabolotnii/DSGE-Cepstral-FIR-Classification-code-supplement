"""Verification protocols for DSGE experiments.

These checks codify the verification standards from the
kunchenko-research-workflow skill (Verification Standard 8 + PATP checks):

    - bootstrap p-value for hybrid > baseline
    - 95% CI for the macro-F1 improvement
    - basis comparison ablation
    - PATP boundary check (best alpha not at 0 or 1)
    - F-matrix conditioning report

Use these BEFORE submitting a paper or pushing a release. They take a
few seconds to run and catch common publication-killer mistakes.

Public API:
    bootstrap_compare(y_true, y_pred_baseline, y_pred_hybrid, R=1000)
        -> {'delta', 'p_value', 'ci_low', 'ci_high', 'significant'}

    verification_report(results, ...) -> str
        Pretty-prints PASS/FAIL for each Verification Standard.
"""

from __future__ import annotations

import numpy as np


def bootstrap_compare(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_hybrid: np.ndarray,
    R: int = 1000,
    metric: str = "macro_f1",
    seed: int = 42,
) -> dict:
    """Paired bootstrap test on macro-F1 (or accuracy) improvement.

    H0: hybrid model gives the same metric as baseline.
    Returns p-value (two-sided), 95% CI, and `significant` flag at p < 0.05.

    R = 1000 is the standard from the NLP paper (matches the
    kunchenko-research-workflow Verification Standard 8). Pass R = 5000+
    for tighter CIs on small test sets.
    """
    from sklearn.metrics import accuracy_score, f1_score
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred_baseline = np.asarray(y_pred_baseline)
    y_pred_hybrid = np.asarray(y_pred_hybrid)
    n = len(y_true)

    if metric == "macro_f1":
        score = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
    elif metric == "accuracy":
        score = accuracy_score
    else:
        raise ValueError(f"Unknown metric '{metric}'.")

    deltas = np.zeros(R)
    for r in range(R):
        idx = rng.integers(0, n, size=n)
        s_b = score(y_true[idx], y_pred_baseline[idx])
        s_h = score(y_true[idx], y_pred_hybrid[idx])
        deltas[r] = s_h - s_b

    point = score(y_true, y_pred_hybrid) - score(y_true, y_pred_baseline)
    # Two-sided p-value: fraction of bootstrap deltas at least as extreme
    # as the observed point estimate, under the null of no difference.
    p_value = float(2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": float(point),
        "p_value": p_value,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "significant": bool(p_value < 0.05 and point > 0),
        "metric": metric,
        "R": R,
    }


def patp_boundary_check(alphas: list[float], scores: list[float],
                        margin: float = 0.05) -> dict:
    """Confirm that the optimal alpha is not at the [0, 1] boundary.

    If alpha_opt sits at 0 or 1 (within `margin`), the corresponding fixed
    discrete basis is sufficient and PATP gives no benefit.

    Returns:
        {'best_alpha', 'at_boundary', 'recommendation'}
    """
    best_idx = int(np.argmax(scores))
    best_alpha = float(alphas[best_idx])
    at_boundary = best_alpha < margin or best_alpha > (1.0 - margin)
    rec = (
        f"alpha_opt = {best_alpha:.2f} is at the boundary; use the "
        f"corresponding fixed basis ({'fractional' if best_alpha < margin else 'integer-power'})."
        if at_boundary else
        f"alpha_opt = {best_alpha:.2f} is interior; PATP gives genuine benefit "
        "over discrete bases."
    )
    return {"best_alpha": best_alpha, "at_boundary": at_boundary, "recommendation": rec}


def conditioning_report(extractor) -> dict:
    """Return per-class cond(F_reg) and a recommendation if poorly conditioned."""
    cond = extractor.conditioning_report
    high = {c: v for c, v in cond.items() if v > 1e6}
    rec = "OK" if not high else (
        f"Classes {list(high)} have cond(F) > 1e6. "
        "Consider increasing `ridge`, decreasing `n`, or trying a smoother basis."
    )
    return {"cond_per_class": cond, "warning": rec}


def verification_report(
    *,
    accuracy_dsge_only: float | None = None,
    accuracy_baseline: float | None = None,
    accuracy_hybrid: float | None = None,
    bootstrap_result: dict | None = None,
    basis_comparison: list[dict] | None = None,
    patp_check: dict | None = None,
    cond_report: dict | None = None,
) -> str:
    """Produce a PASS/FAIL summary aligned with the kunchenko-research-workflow
    Verification Standards 8 (DSGE) and N (PATP).

    Pass any subset of arguments — only the corresponding checks are run.
    """
    lines = ["=== DSGE Verification Report ==="]
    passed = total = 0

    def _check(label: str, condition: bool, detail: str = ""):
        nonlocal passed, total
        total += 1
        passed += int(condition)
        tag = "[PASS]" if condition else "[FAIL]"
        lines.append(f"{tag} {label}{(' — ' + detail) if detail else ''}")

    if accuracy_dsge_only is not None and accuracy_baseline is not None:
        _check(
            "DSGE-only beats trivial baseline",
            accuracy_dsge_only > 0.5 + 1e-6,
            f"acc_dsge = {accuracy_dsge_only:.3f}",
        )

    if accuracy_hybrid is not None and accuracy_baseline is not None:
        _check(
            "Hybrid > baseline",
            accuracy_hybrid > accuracy_baseline,
            f"hybrid = {accuracy_hybrid:.3f} vs baseline = {accuracy_baseline:.3f}",
        )

    if bootstrap_result is not None:
        _check(
            "Bootstrap p < 0.05 (Verification Standard 8)",
            bootstrap_result["significant"],
            f"p = {bootstrap_result['p_value']:.4f}, "
            f"95% CI [{bootstrap_result['ci_low']:+.4f}, {bootstrap_result['ci_high']:+.4f}]",
        )

    if basis_comparison:
        bases = [r["basis"] for r in basis_comparison]
        f1s = [r["macro_f1"] for r in basis_comparison]
        best = bases[int(np.argmax(f1s))]
        _check(
            "At least one basis gives non-trivial F1",
            max(f1s) > 0.5,
            f"best basis = {best} (F1 = {max(f1s):.3f})",
        )

    if patp_check is not None:
        _check(
            "PATP alpha_opt is interior (not at boundary)",
            not patp_check["at_boundary"],
            patp_check["recommendation"],
        )

    if cond_report is not None:
        _check(
            "F-matrix well-conditioned (cond < 1e6 per class)",
            "OK" in cond_report["warning"],
            cond_report["warning"],
        )

    lines.append(f"=== {passed}/{total} checks passed ===")
    return "\n".join(lines)
