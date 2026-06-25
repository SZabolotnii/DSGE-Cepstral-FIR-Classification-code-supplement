"""Definitive equal-budget H2 test on real data — DSGE-11 vs fixed-stat-11.

The spec §7 RQ2 demands an *equal feature budget* comparison. DSGE produces
exactly ``n_classes`` log-MSED features (11 here), one per class — that count
is intrinsic, not a tunable knob. The original `features.spectral_statistics`
exposes only 5 closed-form descriptors, so the headline H2 (DSGE-11 vs
stat-5) is budget-unequal, and forcing k=5 on both sides cripples DSGE by
discarding 6 of its 11 natural reconstruction errors. Neither is the fair
test.

This module builds an **11-dimensional fixed-stat baseline** by adding six
more permutation-invariant across-bin shape descriptors to the original five,
then compares DSGE-11 vs fixed-stat-11 at matched budget on the same
leakage-safe split. Permutation-invariant descriptors are used so the
comparison stays within the "shape of the across-bin distribution" family
that DSGE-profile lives in (the spec's framing of DSGE as an adaptive
generalisation of fixed shape statistics).

The 11 fixed descriptors:
  1 flatness          (geo/arith mean of mag²)
  2 entropy           (Shannon entropy of the bin pmf, normalised)
  3 centroid          (energy-weighted bin index — position-aware; kept for
                       parity with `spectral_statistics`)
  4 skewness          (across-bin, of the log-profile)
  5 kurtosis          (excess, across-bin)
  6 crest             (max − mean of log-profile = peak-to-average, log dB)
  7 p05               (5th percentile of the log-profile)
  8 p50               (median)
  9 p95               (95th percentile)
 10 iqr               (p75 − p25)
 11 top_decile_gap    (mean of top-10% bins − mean of bottom-10% bins)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

_TOOLKIT = Path(__file__).resolve().parents[1] / "toolkit"
if str(_TOOLKIT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT))

from kunchenko_features import DSGEFeatureExtractor  # noqa: E402
from splits import grouped_split  # noqa: E402
from verify_dsge import bootstrap_compare  # noqa: E402

from features import spectral_statistics, _LOG_EPS  # noqa: E402
from real_data import build_real_dataset  # noqa: E402


EXTENDED_NAMES = [
    "flatness", "entropy", "centroid", "skewness", "kurtosis",
    "crest", "p05", "p50", "p95", "iqr", "top_decile_gap",
]


def spectral_statistics_extended(profile: np.ndarray) -> np.ndarray:
    """11 across-bin descriptors per frame (see module docstring)."""
    base5 = spectral_statistics(profile)  # (n, 5)
    # Additional permutation-invariant descriptors on the log-profile.
    p = profile
    mx = p.max(axis=1)
    mean = p.mean(axis=1)
    crest = mx - mean
    p05 = np.percentile(p, 5, axis=1)
    p25 = np.percentile(p, 25, axis=1)
    p50 = np.percentile(p, 50, axis=1)
    p75 = np.percentile(p, 75, axis=1)
    p95 = np.percentile(p, 95, axis=1)
    iqr = p75 - p25
    # Top/bottom decile gap.
    n_bins = p.shape[1]
    k = max(1, n_bins // 10)
    sorted_p = np.sort(p, axis=1)
    bottom = sorted_p[:, :k].mean(axis=1)
    top = sorted_p[:, -k:].mean(axis=1)
    gap = top - bottom
    extra = np.stack([crest, p05, p50, p95, iqr, gap], axis=1)
    return np.concatenate([base5, extra], axis=1)


def _fit_lr(feat, y):
    sc = StandardScaler().fit(feat)
    return sc, LogisticRegression(max_iter=2000, n_jobs=1).fit(sc.transform(feat), y)


def main(seed: int = 2026, bootstrap_R: int = 1000):
    base = Path(__file__).resolve().parent.parent
    out = base / "results" / "real" / "h2_equal_budget.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = build_real_dataset(
            n_windows_per_file=200, win_samples=4096,
            n_fft=256, hop=128, n_time_blocks=20,
            energy_std_thresh=0.8, max_frames_per_class=5000, seed=seed,
        )
        profiles, y, g = ds["profiles"], ds["y"], ds["groups"]
        n_classes = ds["n_classes"]
        train_idx, test_idx = grouped_split(
            profiles.shape[0], groups=g, test_size=0.3, random_state=seed,
        )
        X_tr, X_te = profiles[train_idx], profiles[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Extended fixed-stat → 11 features.
        fs_tr = spectral_statistics_extended(X_tr)
        fs_te = spectral_statistics_extended(X_te)
        n_stat = fs_tr.shape[1]
        sc_s, clf_s = _fit_lr(fs_tr, y_tr)
        yp_stat = clf_s.predict(sc_s.transform(fs_te))
        f1_stat = float(f1_score(y_te, yp_stat, average="macro"))
        acc_stat = float(accuracy_score(y_te, yp_stat))

        # DSGE winner (frozen config from corrected h2.json) → n_classes feats.
        winner = json.loads((base / "results" / "real" / "h2.json").read_text())["winner"]
        if winner["kind"] == "patp":
            ext_kwargs = dict(basis="patp", n=int(winner["n"]),
                              alpha=float(winner["alpha"]), ridge=0.01)
        else:
            ext_kwargs = dict(basis=winner["basis"], n=3, ridge=0.01)
        sc_d = StandardScaler().fit(X_tr)
        ext = DSGEFeatureExtractor(**ext_kwargs)
        fd_tr = ext.fit_transform(sc_d.transform(X_tr), y_tr)
        fd_te = ext.transform(sc_d.transform(X_te))
        n_dsge = fd_tr.shape[1]
        sc_dl, clf_dl = _fit_lr(fd_tr, y_tr)
        yp_dsge = clf_dl.predict(sc_dl.transform(fd_te))
        f1_dsge = float(f1_score(y_te, yp_dsge, average="macro"))
        acc_dsge = float(accuracy_score(y_te, yp_dsge))

        bs = bootstrap_compare(
            y_true=y_te, y_pred_baseline=yp_stat, y_pred_hybrid=yp_dsge,
            R=bootstrap_R, metric="macro_f1", seed=seed,
        )
        verdict = "PASS" if (bs["significant"] and bs["delta"] > 0) else "FAIL"

        # --- Decisive H4★ at strong baseline: does DSGE still add anything on
        # top of the FULL 11-dim fixed-stat + PCA-position pipeline? If not,
        # DSGE-profile is fully subsumed on real data. ---
        from sklearn.decomposition import PCA
        sc_pos = StandardScaler().fit(X_tr)
        pca = PCA(n_components=20, random_state=seed)
        pos_tr = pca.fit_transform(sc_pos.transform(X_tr))
        pos_te = pca.transform(sc_pos.transform(X_te))
        base_tr = np.concatenate([fs_tr, pos_tr], axis=1)
        base_te = np.concatenate([fs_te, pos_te], axis=1)
        full_tr = np.concatenate([fd_tr, fs_tr, pos_tr], axis=1)
        full_te = np.concatenate([fd_te, fs_te, pos_te], axis=1)
        sc_b, clf_b = _fit_lr(base_tr, y_tr)
        sc_f, clf_f = _fit_lr(full_tr, y_tr)
        yp_b = clf_b.predict(sc_b.transform(base_te))
        yp_f = clf_f.predict(sc_f.transform(full_te))
        f1_b = float(f1_score(y_te, yp_b, average="macro"))
        f1_f = float(f1_score(y_te, yp_f, average="macro"))
        bs_h4s = bootstrap_compare(
            y_true=y_te, y_pred_baseline=yp_b, y_pred_hybrid=yp_f,
            R=bootstrap_R, metric="macro_f1", seed=seed + 1,
        )
        h4s_verdict = "PASS" if (bs_h4s["significant"] and bs_h4s["delta"] > 0) else "FAIL"

        result = dict(
            note="Definitive equal-budget H2: DSGE (intrinsic n_classes feats) "
                 "vs fixed-stat extended to 11 permutation-invariant shape "
                 "descriptors.",
            dsge_config=ext_kwargs,
            n_features=dict(dsge=int(n_dsge), fixed_stat=int(n_stat)),
            extended_stat_names=EXTENDED_NAMES,
            dsge=dict(test_macro_f1=f1_dsge, test_accuracy=acc_dsge),
            fixed_stat=dict(test_macro_f1=f1_stat, test_accuracy=acc_stat),
            bootstrap=bs,
            verdict=verdict,
            h4star_strong=dict(
                note="DSGE + 11-dim fixed-stat + PCA(20) vs 11-dim fixed-stat "
                     "+ PCA(20). Does DSGE add anything on the strongest "
                     "non-DSGE pipeline?",
                baseline_macro_f1=f1_b,
                full_macro_f1=f1_f,
                bootstrap=bs_h4s,
                verdict=h4s_verdict,
            ),
        )
        out.write_text(json.dumps(result, indent=2))

        print(f"\n=== Definitive equal-budget H2 (DSGE {n_dsge}-dim vs "
              f"fixed-stat {n_stat}-dim) ===")
        print(f"DSGE ({winner['kind']} {ext_kwargs}): acc={acc_dsge:.4f} f1={f1_dsge:.4f}")
        print(f"Fixed-stat (11 shape descriptors): acc={acc_stat:.4f} f1={f1_stat:.4f}")
        print(f"Δ = {bs['delta']:+.4f}  p = {bs['p_value']:.3e}  "
              f"CI = [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]")
        print(f"Verdict (equal budget k≈11): {verdict}")
        print(f"\n--- Decisive H4★ (strong 11-dim baseline) ---")
        print(f"Baseline (stat11+pos20): f1={f1_b:.4f}  Full (+DSGE): f1={f1_f:.4f}")
        print(f"Δ = {bs_h4s['delta']:+.4f}  p = {bs_h4s['p_value']:.3e}  "
              f"CI = [{bs_h4s['ci_low']:+.4f}, {bs_h4s['ci_high']:+.4f}]  → {h4s_verdict}")
        print(f"Saved: {out}")
        return result


if __name__ == "__main__":
    main()
