"""Stats-as-basis DSGE — does feeding the *winning* descriptors into the
Kunchenko apparatus recover the lift that the raw-value power basis lost?

Two constructions (Zabolotnii's proposal):

  C1 — "DSGE over descriptors": take the winning shape-descriptor vector S(x)
       as the object decomposed in the generating-element space; DSGE applies
       its power/fractional basis component-wise to the standardised
       descriptors and solves F·K=B per class → log-MSED reconstruction-error
       features. (Implemented with the toolkit's DSGEFeatureExtractor on S.)

  C2 — "descriptors literally as the basis functionals": φ_i = S_i, generating
       element via leave-one-descriptor-out reconstruction. Per class the
       optimal weights solve F_c·K_c=B_c with F_c = within-class descriptor
       covariance; the summed reconstruction residual is, up to algebra, the
       per-class Mahalanobis quadratic form (x−μ_c)ᵀ Σ_c⁻¹ (x−μ_c). We compute
       that directly. It is therefore mathematically ≈ regularised QDA in
       descriptor space — so we include sklearn QDA as an honesty check: C2 is
       a genuine new method only if it does something QDA does not.

Baselines: LR(S) (the current fixed-stat winner) and QDA(S).
All methods map the d-dim descriptor vector → n_classes reconstruction/score
features (C1, C2) or use S directly (LR, QDA). Bootstrap vs LR(S) and C2 vs QDA.

Run on: RF DroneRF (extended 11 spectral descriptors, 11 classes — has
head-room) and drilling (33 descriptors, 2 classes — LR already ≈0.98, a
control for "rebranding").
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats as sps
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

_TOOLKIT = Path(__file__).resolve().parents[1] / "toolkit"
if str(_TOOLKIT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT))
from dsge_pipeline import run_dsge_pipeline, patp_alpha_sweep  # noqa: E402
from kunchenko_features import DSGEFeatureExtractor  # noqa: E402
from splits import grouped_split  # noqa: E402
from verify_dsge import bootstrap_compare  # noqa: E402

from baseline_pack import baseline_pack_predict  # A1 comparators (C0/RDA/diag-QDA/Ghosh-GAM)

SEED = 2026
ALPHA_GRID = np.linspace(0.0, 1.0, 11)


def _lr():
    return LogisticRegression(max_iter=2000, n_jobs=1)


def _fit_lr(X, y):
    sc = StandardScaler().fit(X)
    return sc, _lr().fit(sc.transform(X), y)


def _f1(yt, yp):
    return float(f1_score(yt, yp, average="macro"))


def _fmt_p(p):
    return f"{p:.3e}" if p < 1e-4 else f"{p:.4f}"


# ---------------- Construction 1: DSGE power/frac basis over S ------------- #

def select_basis_on_S(S_tr, y_tr, g_tr, seed):
    inner, val = grouped_split(S_tr.shape[0], groups=g_tr, test_size=0.25,
                               random_state=seed + 10)
    Xtr, ytr, Xv, yv = S_tr[inner], y_tr[inner], S_tr[val], y_tr[val]
    rows = {}
    for basis in ["polynomial", "fractional", "robust"]:
        r = run_dsge_pipeline(X_train=Xtr, y_train=ytr, X_test=Xv, y_test=yv,
                              basis=basis, n=3, ridge=0.01, standardize=True,
                              classifier=_lr())
        rows[basis] = float(r["main"].macro_f1)
    sweep = patp_alpha_sweep(Xtr, ytr, Xv, yv, n=3, ridge=0.01,
                             alphas=ALPHA_GRID, classifier=_lr(), standardize=True)
    best = max(rows, key=rows.get)
    if sweep["best_macro_f1"] > rows[best]:
        return dict(kind="patp", basis="patp", alpha=float(sweep["best_alpha"]), n=3)
    return dict(kind="discrete", basis=best, alpha=None, n=3)


def c1_dsge_over_stats(S_tr, y_tr, g_tr, S_te, seed):
    cfg = select_basis_on_S(S_tr, y_tr, g_tr, seed)
    kw = (dict(basis="patp", n=cfg["n"], alpha=cfg["alpha"], ridge=0.01)
          if cfg["kind"] == "patp" else dict(basis=cfg["basis"], n=3, ridge=0.01))
    sc = StandardScaler().fit(S_tr)
    ext = DSGEFeatureExtractor(**kw)
    f_tr = ext.fit_transform(sc.transform(S_tr), y_tr)
    f_te = ext.transform(sc.transform(S_te))
    return f_tr, f_te, cfg


# ---------------- Construction 2: stats-as-basis Kunchenko = Mahalanobis --- #

def c2_kunchenko_mahalanobis(S_tr, y_tr, S_te, ridge=1e-2):
    """Per-class reconstruction error with φ_i = S_i and leave-one-out
    generating element. Summing the per-descriptor residuals yields the
    per-class Mahalanobis quadratic form under the within-class descriptor
    covariance F_c (Tikhonov-regularised) — computed directly here.

    Returns (feat_tr, feat_te) of shape (n, n_classes): D²_c per class.
    """
    classes = np.unique(y_tr)
    sc = StandardScaler().fit(S_tr)
    Z_tr = sc.transform(S_tr)
    Z_te = sc.transform(S_te)
    d = Z_tr.shape[1]
    feats_tr = np.empty((Z_tr.shape[0], len(classes)))
    feats_te = np.empty((Z_te.shape[0], len(classes)))
    for ci, c in enumerate(classes):
        Zc = Z_tr[y_tr == c]
        mu = Zc.mean(0)
        F = np.cov(Zc, rowvar=False) + ridge * np.eye(d)   # within-class F_c
        Finv = np.linalg.inv(F)
        dtr = Z_tr - mu
        dte = Z_te - mu
        feats_tr[:, ci] = np.einsum("ni,ij,nj->n", dtr, Finv, dtr)
        feats_te[:, ci] = np.einsum("ni,ij,nj->n", dte, Finv, dte)
    # log for scale stability (mirrors log-MSED)
    return np.log(feats_tr + 1e-9), np.log(feats_te + 1e-9)


# ---------------- generic comparison ---------------- #

def compare(S_tr, y_tr, g_tr, S_te, y_te, n_classes, label, R=1000, seed=SEED):
    # baseline LR(S) — the current fixed-stat winner
    sc_b, clf_b = _fit_lr(S_tr, y_tr)
    yp_lr = clf_b.predict(sc_b.transform(S_te))
    f1_lr = _f1(y_te, yp_lr)
    # QDA(S) — honesty check for C2
    sc_q = StandardScaler().fit(S_tr)
    qda = QuadraticDiscriminantAnalysis(reg_param=0.05).fit(sc_q.transform(S_tr), y_tr)
    yp_qda = qda.predict(sc_q.transform(S_te))
    f1_qda = _f1(y_te, yp_qda)
    # C1 — DSGE power/frac over S
    c1_tr, c1_te, c1_cfg = c1_dsge_over_stats(S_tr, y_tr, g_tr, S_te, seed)
    sc1, clf1 = _fit_lr(c1_tr, y_tr)
    yp_c1 = clf1.predict(sc1.transform(c1_te))
    f1_c1 = _f1(y_te, yp_c1)
    # C2 — Kunchenko-Mahalanobis over S
    c2_tr, c2_te = c2_kunchenko_mahalanobis(S_tr, y_tr, S_te)
    sc2, clf2 = _fit_lr(c2_tr, y_tr)
    yp_c2 = clf2.predict(sc2.transform(c2_te))
    f1_c2 = _f1(y_te, yp_c2)

    def bs(yp_base, yp_new, sd):
        r = bootstrap_compare(y_true=y_te, y_pred_baseline=yp_base,
                              y_pred_hybrid=yp_new, R=R, metric="macro_f1", seed=sd)
        return dict(delta=r["delta"], p=r["p_value"], ci=[r["ci_low"], r["ci_high"]],
                    sig=bool(r["significant"] and r["delta"] > 0))

    # A1 baseline-pack — C0 / RDA / diag-QDA / Ghosh-GAM on the same descriptors S
    pack_yp, pack_cfg = baseline_pack_predict(S_tr, y_tr, S_te, seed=seed)
    pack_f1 = {name: _f1(y_te, yp) for name, yp in pack_yp.items()}
    # bootstrap each pack member vs LR (and C2 vs each, to position C2 against the pack)
    pack_vs_lr = {name: bs(yp_lr, yp, seed + 3 + i)
                  for i, (name, yp) in enumerate(pack_yp.items())}
    c2_vs_pack = {name: bs(yp, yp_c2, seed + 7 + i)
                  for i, (name, yp) in enumerate(pack_yp.items())}

    out = dict(
        label=label, n_classes=int(n_classes), n_descriptors=int(S_tr.shape[1]),
        n_train=int(S_tr.shape[0]), n_test=int(S_te.shape[0]),
        c1_config=c1_cfg, baseline_pack_config=pack_cfg,
        macro_f1=dict(LR_stats=f1_lr, QDA_stats=f1_qda,
                      C1_dsge_over_stats=f1_c1, C2_kunchenko_mahalanobis=f1_c2,
                      **{f"pack_{k}": v for k, v in pack_f1.items()}),
        C1_vs_LR=bs(yp_lr, yp_c1, seed),
        C2_vs_LR=bs(yp_lr, yp_c2, seed + 1),
        C2_vs_QDA=bs(yp_qda, yp_c2, seed + 2),
        pack_vs_LR=pack_vs_lr,
        C2_vs_pack=c2_vs_pack,
    )
    return out


def print_block(o):
    m = o["macro_f1"]
    print(f"\n=== {o['label']} ({o['n_classes']} classes, {o['n_descriptors']} descriptors, "
          f"{o['n_train']}/{o['n_test']}) ===")
    print(f"  LR(stats)               f1={m['LR_stats']:.4f}   [current winner]")
    print(f"  QDA(stats)              f1={m['QDA_stats']:.4f}   [honesty check for C2]")
    print(f"  C1 DSGE-over-stats      f1={m['C1_dsge_over_stats']:.4f}   cfg={o['c1_config']}")
    print(f"  C2 Kunchenko-Mahal      f1={m['C2_kunchenko_mahalanobis']:.4f}")
    print("  -- A1 baseline-pack (same descriptors, equal budget) --")
    for name in ("C0_fixed_rule", "RDA", "diag_QDA", "Ghosh_GAM"):
        f1 = m.get(f"pack_{name}")
        if f1 is not None:
            print(f"  {name:<22} f1={f1:.4f}")
    print(f"  pack config: {o.get('baseline_pack_config')}")
    for k in ("C1_vs_LR", "C2_vs_LR", "C2_vs_QDA"):
        b = o[k]
        print(f"  {k:<10}: Δ={b['delta']:+.4f} p={_fmt_p(b['p'])} "
              f"CI=[{b['ci'][0]:+.4f},{b['ci'][1]:+.4f}] {'PASS' if b['sig'] else 'ns'}")
    for name, b in o.get("C2_vs_pack", {}).items():
        print(f"  C2_vs_{name:<16}: Δ={b['delta']:+.4f} p={_fmt_p(b['p'])} "
              f"CI=[{b['ci'][0]:+.4f},{b['ci'][1]:+.4f}] {'PASS' if b['sig'] else 'ns'}")


# ---------------- dataset builders ---------------- #

def build_rf():
    from run_real import build_real_dataset
    from h2_equal_budget import spectral_statistics_extended
    ds = build_real_dataset(n_windows_per_file=200, win_samples=4096, n_fft=256,
                            hop=128, n_time_blocks=20, energy_std_thresh=0.8,
                            max_frames_per_class=5000, seed=SEED)
    prof, y, g = ds["profiles"], ds["y"], ds["groups"]
    S = spectral_statistics_extended(prof)
    tr, te = grouped_split(prof.shape[0], groups=g, test_size=0.3, random_state=SEED)
    return (S[tr], y[tr], g[tr], S[te], y[te], ds["n_classes"], "RF DroneRF 2G")


def build_drilling():
    from run_drilling import (load_volve, load_forge_58_32, make_windows,
                              balance_by_groups, shape_features)
    volve = load_volve(); forge = load_forge_58_32()
    pV, gV, sV = make_windows(volve, 0, 0)
    pF, gF, sF = make_windows(forge, 1, 1_000_000)
    rng = np.random.default_rng(SEED)
    cap = min(len(pV), len(pF))
    pF, gF, sF = balance_by_groups(pF, gF, sF, cap, rng)
    pV, gV, sV = balance_by_groups(pV, gV, sV, cap, rng)
    prof = np.concatenate([pV, pF]); y = np.concatenate([np.zeros(len(pV), int), np.ones(len(pF), int)])
    g = np.concatenate([gV, gF])
    S = shape_features(prof)
    tr, te = grouped_split(prof.shape[0], groups=g, test_size=0.3, random_state=SEED)
    return (S[tr], y[tr], g[tr], S[te], y[te], 2, "Drilling Volve-vs-FORGE")


def main():
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "results" / "stats_basis"
    out_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = {}
        for builder in (build_rf, build_drilling):
            S_tr, y_tr, g_tr, S_te, y_te, ncl, label = builder()
            o = compare(S_tr, y_tr, g_tr, S_te, y_te, ncl, label)
            print_block(o)
            results[label] = o
        (out_dir / "stats_basis.json").write_text(json.dumps(results, indent=2))
        print(f"\nSaved: {out_dir / 'stats_basis.json'}")


if __name__ == "__main__":
    main()
