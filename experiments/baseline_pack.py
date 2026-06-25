"""A1 baseline-pack for Paper 1 (C2) Table 1 — the comparators the ROADMAP
flags as the critical empirical gap before submission.

Four leakage-safe, equal-budget baselines on the *same* descriptor vector S(x)
that LR / QDA / C2 receive:

  C0       — argmin_c D²_c with the same per-class Tikhonov correlant F_c, no
             learned head: the fixed-rule ablation of C2 (isolates the head's
             contribution). If C2 ≈ C0, the learned stacking adds nothing.

  RDA      — Friedman (1989) regularized discriminant analysis: shrink each
             class covariance toward the pooled one (λ) and toward a scaled
             identity (γ); Gaussian plug-in rule. (λ, γ) picked on an inner split.

  diag-QDA — Bickel & Levina (2004) diagonal/naive-Gaussian per-class rule
             (= GaussianNB): the n<p-safe degenerate QDA.

  Ghosh-GAM — Ghosh et al. (2025, arXiv 2402.08283): feed the vector of per-class
             Mahalanobis distances D²_c into a logistic generalized additive
             model. Approximated faithfully with a spline expansion of the D²_c
             features → multinomial logistic (logistic-link GAM). Closest prior
             art; the comparator a reviewer who knows Ghosh 2025 will demand.

All functions are pure numpy/sklearn (no toolkit / no external data), so they are
unit-testable standalone (`python baseline_pack.py` runs a synthetic self-test).
Each `*_predict` returns hard test predictions, so the caller scores macro-F1 and
bootstraps against LR with the *same* protocol as `stats_basis.compare`.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import SplineTransformer, StandardScaler

RIDGE = 1e-2


# --------------------------------------------------------------------------- #
# Shared per-class Mahalanobis (the same F_c C2 uses), for C0 and Ghosh-GAM.
# --------------------------------------------------------------------------- #
def per_class_mahalanobis(S_tr, y_tr, S_te, ridge=RIDGE):
    """Return (D2_tr, D2_te, classes): per-class Tikhonov Mahalanobis D²_c on
    standardized descriptors. Mirrors stats_basis.c2_kunchenko_mahalanobis but
    returns the *raw* D²_c (no log) so C0's argmin and Ghosh's GAM see distances.
    """
    classes = np.unique(y_tr)
    sc = StandardScaler().fit(S_tr)
    Z_tr, Z_te = sc.transform(S_tr), sc.transform(S_te)
    d = Z_tr.shape[1]
    D2_tr = np.empty((Z_tr.shape[0], len(classes)))
    D2_te = np.empty((Z_te.shape[0], len(classes)))
    for ci, c in enumerate(classes):
        Zc = Z_tr[y_tr == c]
        mu = Zc.mean(0)
        F = np.cov(Zc, rowvar=False) + ridge * np.eye(d)
        Finv = np.linalg.inv(F)
        dtr, dte = Z_tr - mu, Z_te - mu
        D2_tr[:, ci] = np.einsum("ni,ij,nj->n", dtr, Finv, dtr)
        D2_te[:, ci] = np.einsum("ni,ij,nj->n", dte, Finv, dte)
    return D2_tr, D2_te, classes


# --------------------------------------------------------------------------- #
# C0 — fixed-rule argmin_c D²_c (no learned head)
# --------------------------------------------------------------------------- #
def c0_fixed_rule_predict(S_tr, y_tr, S_te, ridge=RIDGE):
    _, D2_te, classes = per_class_mahalanobis(S_tr, y_tr, S_te, ridge)
    return classes[np.argmin(D2_te, axis=1)]


# --------------------------------------------------------------------------- #
# RDA — Friedman (1989) regularized discriminant analysis
# --------------------------------------------------------------------------- #
def _rda_fit(Z_tr, y_tr, lam, gam):
    """Fit Friedman RDA covariances. Returns per-class (mu, inv_cov, logdet, prior)."""
    classes = np.unique(y_tr)
    d = Z_tr.shape[1]
    # pooled covariance
    pooled = np.zeros((d, d))
    n = len(y_tr)
    for c in classes:
        Zc = Z_tr[y_tr == c]
        pooled += (len(Zc) - 1) * np.cov(Zc, rowvar=False)
    pooled /= (n - len(classes))
    params = {}
    for c in classes:
        Zc = Z_tr[y_tr == c]
        Sc = np.cov(Zc, rowvar=False)
        Sc_lam = (1 - lam) * Sc + lam * pooled            # shrink toward pooled
        trace_term = np.trace(Sc_lam) / d
        Sigma = (1 - gam) * Sc_lam + gam * trace_term * np.eye(d)  # toward I
        sign, logdet = np.linalg.slogdet(Sigma)
        params[c] = (Zc.mean(0), np.linalg.inv(Sigma), logdet, len(Zc) / n)
    return classes, params


def _rda_predict(Z, classes, params):
    scores = np.empty((Z.shape[0], len(classes)))
    for ci, c in enumerate(classes):
        mu, inv, logdet, prior = params[c]
        diff = Z - mu
        maha = np.einsum("ni,ij,nj->n", diff, inv, diff)
        scores[:, ci] = -0.5 * maha - 0.5 * logdet + np.log(prior)  # Gaussian plug-in
    return classes[np.argmax(scores, axis=1)]


def rda_predict(S_tr, y_tr, S_te, seed=2026, grid=None):
    """Friedman RDA with (λ, γ) chosen on an inner stratified split (nested
    selection; the outer test is never touched for tuning)."""
    if grid is None:
        grid = [(l, g) for l in (0.0, 0.25, 0.5, 0.75, 1.0)
                for g in (0.0, 0.1, 0.25, 0.5)]
    sc = StandardScaler().fit(S_tr)
    Z_tr, Z_te = sc.transform(S_tr), sc.transform(S_te)
    rng = np.random.default_rng(seed)
    # inner stratified split
    idx_inner, idx_val = [], []
    for c in np.unique(y_tr):
        ci = np.where(y_tr == c)[0]
        rng.shuffle(ci)
        cut = max(1, int(0.75 * len(ci)))
        idx_inner += list(ci[:cut]); idx_val += list(ci[cut:])
    idx_inner, idx_val = np.array(idx_inner), np.array(idx_val)
    from sklearn.metrics import f1_score
    best, best_f1 = grid[0], -1.0
    for lam, gam in grid:
        try:
            classes, params = _rda_fit(Z_tr[idx_inner], y_tr[idx_inner], lam, gam)
            yp = _rda_predict(Z_tr[idx_val], classes, params)
            f1 = f1_score(y_tr[idx_val], yp, average="macro")
        except np.linalg.LinAlgError:
            continue
        if f1 > best_f1:
            best, best_f1 = (lam, gam), f1
    classes, params = _rda_fit(Z_tr, y_tr, *best)
    return _rda_predict(Z_te, classes, params), {"lambda": best[0], "gamma": best[1]}


# --------------------------------------------------------------------------- #
# diagonal-QDA — Bickel & Levina (2004); GaussianNB is exactly per-class diag.
# --------------------------------------------------------------------------- #
def diag_qda_predict(S_tr, y_tr, S_te):
    sc = StandardScaler().fit(S_tr)
    clf = GaussianNB().fit(sc.transform(S_tr), y_tr)
    return clf.predict(sc.transform(S_te))


# --------------------------------------------------------------------------- #
# Ghosh-GAM — per-class Mahalanobis D²_c -> spline (GAM) -> multinomial logistic
# --------------------------------------------------------------------------- #
def ghosh_gam_predict(S_tr, y_tr, S_te, ridge=RIDGE, n_knots=5, degree=3, seed=2026):
    """Faithful approximation of Ghosh et al. (2025): the feature vector of
    per-class Mahalanobis distances D²_c is fed to a logistic-link generalized
    additive model. The additive smooth of each D²_c is realised with a B-spline
    basis expansion (per feature) followed by a multinomial logistic head."""
    D2_tr, D2_te, _ = per_class_mahalanobis(S_tr, y_tr, S_te, ridge)
    # log-stabilize the chi-square-like distances before the smooth (Ghosh use a
    # nonparametric smooth; the spline absorbs monotone reparametrisation either way)
    spl = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)
    sc = StandardScaler()
    Phi_tr = sc.fit_transform(spl.fit_transform(D2_tr))
    Phi_te = sc.transform(spl.transform(D2_te))
    clf = LogisticRegression(max_iter=2000, n_jobs=1).fit(Phi_tr, y_tr)
    return clf.predict(Phi_te)


# --------------------------------------------------------------------------- #
# convenience: all four at once
# --------------------------------------------------------------------------- #
def baseline_pack_predict(S_tr, y_tr, S_te, seed=2026):
    """Return {name: y_pred_test} for C0, RDA, diag-QDA, Ghosh-GAM."""
    rda_yp, rda_cfg = rda_predict(S_tr, y_tr, S_te, seed=seed)
    return {
        "C0_fixed_rule": c0_fixed_rule_predict(S_tr, y_tr, S_te),
        "RDA": rda_yp,
        "diag_QDA": diag_qda_predict(S_tr, y_tr, S_te),
        "Ghosh_GAM": ghosh_gam_predict(S_tr, y_tr, S_te, seed=seed),
    }, {"RDA_config": rda_cfg}


# --------------------------------------------------------------------------- #
# standalone synthetic self-test (no external data / no toolkit)
# --------------------------------------------------------------------------- #
def _self_test():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(2026)
    # 3 Gaussian classes in 6-d with distinct means + covariances (QDA-friendly)
    n, d, K = 400, 6, 3
    Xs, ys = [], []
    for c in range(K):
        mu = rng.normal(c * 1.2, 0.3, size=d)
        A = rng.normal(0, 1, size=(d, d))
        cov = A @ A.T / d + np.eye(d) * (0.5 + 0.4 * c)
        Xs.append(rng.multivariate_normal(mu, cov, size=n)); ys.append(np.full(n, c))
    X, y = np.vstack(Xs), np.concatenate(ys)
    perm = rng.permutation(len(y)); X, y = X[perm], y[perm]
    cut = len(y) // 2
    S_tr, y_tr, S_te, y_te = X[:cut], y[:cut], X[cut:], y[cut:]

    preds, cfg = baseline_pack_predict(S_tr, y_tr, S_te)
    print("Self-test — 3 Gaussian classes, d=6 (chance macro-F1 ≈ 0.33):")
    for name, yp in preds.items():
        f1 = f1_score(y_te, yp, average="macro")
        ok = "OK" if f1 > 0.5 else "LOW"
        print(f"  {name:<16} macro-F1={f1:.3f}  [{ok}]")
    print(f"  RDA config selected: {cfg['RDA_config']}")
    # sanity: every predictor beats chance on this separable problem
    assert all(f1_score(y_te, yp, average="macro") > 0.5 for yp in preds.values()), \
        "a baseline failed to beat chance on a separable synthetic problem"
    print("  all baselines beat chance — pack wired correctly.")


if __name__ == "__main__":
    _self_test()
