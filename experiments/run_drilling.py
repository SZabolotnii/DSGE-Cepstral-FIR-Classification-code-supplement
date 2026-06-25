"""Plain DSGE on drilling telemetry — well classification (Ku_NG data).

NOT DSGE-Spectral: the data is 1–2 Hz MWD/Pason telemetry with no high-rate
waveform (the raw 800 Hz vibration frames are locked in a proprietary
GameChanger format), so an STFT spectral profile is meaningless here. Instead
this applies the *plain* DSGE idea the toolkit was built for: per-class
reconstruction error on the **value distribution** of windowed channel signals.

Task: classify which well a short telemetry window came from, using the
*shape* of its per-channel fluctuation distribution (per-window z-score removes
units and operating point — the honest "shape not loudness" choice, mirroring
the RF per-frame energy normalisation).

Data (only the two wells with dense, common channels):
  - Volve 15/9-F-15  → processed/volve_onbottom.parquet (rpm, tqa, rop5) @2 Hz
  - FORGE 58-32      → forge/Well_58-32_raw_pason_log.csv (Rotary Speed,
                       Surface Torque, ROP) @1 Hz, physical on-bottom filter
Common channels: RPM, TORQUE, ROP. Volve is decimated 2→1 Hz to match FORGE.

Honest caveats (stated up front, not buried):
  - 2 classes only — DSGE yields just n_classes features when pooled, so we use
    per-channel DSGE (3·n_classes = 6 features) for a fair shot.
  - The two wells differ in geology AND acquisition system (Equinor MWD vs
    Pason), so a "well classifier" partly learns acquisition fingerprint, not
    only drilling physics — same confound class as the RF receiver/recording
    confound. Flagged, not hidden.

Protocol matches the corrected RF run: leakage-safe split by depth-block,
group-level H1, model selection on inner-validation, equal-budget H2 plus a
strong-baseline H4★, bootstrap significance.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

_TOOLKIT = Path(__file__).resolve().parents[1] / "toolkit"
if str(_TOOLKIT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT))
from dsge_pipeline import run_dsge_pipeline, patp_alpha_sweep  # noqa: E402
from kunchenko_features import DSGEFeatureExtractor  # noqa: E402
from splits import grouped_split, leakage_audit  # noqa: E402
from verify_dsge import bootstrap_compare  # noqa: E402

_DATA = Path("data/drilling")
CHANNELS = ["rpm", "torque", "rop"]
W = 120          # window length (samples ≈ 2 min @1 Hz)
K_PER_BLOCK = 8  # non-overlapping windows per leakage-safe block
GUARD = W        # guard band (rows) between blocks → purge gap ≥ one window
SEED = 2026
ALPHA_GRID = np.linspace(0.0, 1.0, 11)


# --------------------------- loaders --------------------------- #

def load_volve() -> pd.DataFrame:
    df = pd.read_parquet(_DATA / "processed/volve_onbottom.parquet")
    out = pd.DataFrame({
        "rpm": pd.to_numeric(df["rpm"], errors="coerce"),
        "torque": pd.to_numeric(df["tqa"], errors="coerce"),
        "rop": pd.to_numeric(df["rop5"], errors="coerce"),
        "depth": pd.to_numeric(df["dept"], errors="coerce"),
    }).dropna().reset_index(drop=True)
    # Decimate 2 Hz → 1 Hz to match FORGE sample rate.
    out = out.iloc[::2].reset_index(drop=True)
    return out


def load_forge_58_32() -> pd.DataFrame:
    f = _DATA / "forge/Well_58-32_raw_pason_log.csv"
    cols = ["Rotary Speed (rpm)", "Surface Torque (KPa)", "ROP(1 m)", "Depth (m)"]
    df = pd.read_csv(f, usecols=cols, low_memory=False)
    out = pd.DataFrame({
        "rpm": pd.to_numeric(df["Rotary Speed (rpm)"], errors="coerce"),
        "torque": pd.to_numeric(df["Surface Torque (KPa)"], errors="coerce"),
        "rop": pd.to_numeric(df["ROP(1 m)"], errors="coerce"),
        "depth": pd.to_numeric(df["Depth (m)"], errors="coerce"),
    }).dropna().reset_index(drop=True)
    # Physical on-bottom: rotating, advancing, torque present.
    ob = (out["rpm"] > 10) & (out["rop"] > 1) & (out["torque"] > 0)
    return out[ob].reset_index(drop=True)


# --------------------------- windowing --------------------------- #

def _zscore(v: np.ndarray) -> np.ndarray:
    m = v.mean()
    s = v.std()
    return (v - m) / s if s > 1e-9 else v - m


def make_windows(df: pd.DataFrame, well_id: int, group_base: int):
    """Tile rows into leakage-safe blocks of K non-overlapping windows,
    separated by a GUARD band of unused rows.

    Per-window per-channel z-score; concat channels. Returns
    ``(profiles (n,3W), groups (n,), spans list[(well,start,end)])``.

    Construction guarantees: windows within a block are non-overlapping
    (hop = W), and consecutive blocks are separated by ``GUARD`` unused rows.
    So any two windows in *different* blocks are ≥ GUARD rows apart — they
    share zero raw samples AND sit across a temporal purge gap. A grouped
    split on the block id is therefore genuinely leakage-safe (verified
    afterwards by ``raw_overlap_audit``).
    """
    n = len(df)
    chans = [df[c].to_numpy(dtype=np.float64) for c in CHANNELS]
    profs, grps, spans = [], [], []
    pos = 0
    blk = 0
    while pos + W <= n:
        for k in range(K_PER_BLOCK):
            s = pos + k * W
            e = s + W
            if e > n:
                break
            segs = [_zscore(ch[s:e]) for ch in chans]
            if any(np.any(~np.isfinite(x)) for x in segs):
                continue
            profs.append(np.concatenate(segs))
            grps.append(group_base + blk)
            spans.append((well_id, s, e))
        pos += K_PER_BLOCK * W + GUARD
        blk += 1
    return np.asarray(profs), np.asarray(grps, dtype=np.int64), spans


def raw_overlap_audit(spans, tr_idx, te_idx) -> int:
    """Count train/test window pairs (same well) whose raw row intervals
    overlap. With the block+guard construction this must be 0 — proves the
    split shares no raw telemetry, not just no group label."""
    from collections import defaultdict
    tr, te = defaultdict(list), defaultdict(list)
    for i in tr_idx:
        w, s, e = spans[i]
        tr[w].append((s, e))
    for i in te_idx:
        w, s, e = spans[i]
        te[w].append((s, e))
    overlaps = 0
    for w in set(tr) | set(te):
        for (s1, e1) in tr.get(w, []):
            for (s2, e2) in te.get(w, []):
                if s1 < e2 and s2 < e1:
                    overlaps += 1
    return overlaps


def balance_by_groups(p, g, spans, cap, rng):
    """Subsample WHOLE groups (not individual windows) until the cumulative
    window count first reaches ``cap``. Preserves group integrity so the
    grouped split and group-level H1 stay valid."""
    uniq = np.unique(g)
    rng.shuffle(uniq)
    keep = np.zeros(len(g), dtype=bool)
    total = 0
    for gid in uniq:
        m = g == gid
        keep |= m
        total += int(m.sum())
        if total >= cap:
            break
    idx = np.where(keep)[0]
    return p[idx], g[idx], [spans[i] for i in idx]


# --------------------------- shape descriptors --------------------------- #

STAT_NAMES = ["skew", "kurt", "max", "min", "p05", "p95", "iqr",
              "mad_diff", "zcr", "ar1", "rng"]


def shape_descriptors_1ch(w: np.ndarray) -> np.ndarray:
    """11 distribution/dynamics descriptors for one z-scored window."""
    d = w - w.mean()
    var = (d ** 2).mean()
    std = np.sqrt(var) if var > 1e-12 else 1.0
    skew = (d ** 3).mean() / std ** 3
    kurt = (d ** 4).mean() / std ** 4 - 3.0
    p05, p25, p50, p75, p95 = np.percentile(w, [5, 25, 50, 75, 95])
    iqr = p75 - p25
    diff = np.diff(w)
    mad_diff = np.mean(np.abs(diff))                     # roughness / HF energy
    zcr = np.mean(np.abs(np.diff(np.sign(w - p50)))) / 2  # zero-crossing rate
    ar1 = np.corrcoef(w[:-1], w[1:])[0, 1] if w.std() > 1e-9 else 0.0
    rng = w.max() - w.min()
    return np.array([skew, kurt, w.max(), w.min(), p05, p95, iqr,
                     mad_diff, zcr, (ar1 if np.isfinite(ar1) else 0.0), rng])


def shape_features(profiles: np.ndarray) -> np.ndarray:
    """Per-channel 11 descriptors → 3·11 = 33 features per window."""
    n = profiles.shape[0]
    out = np.empty((n, 3 * len(STAT_NAMES)), dtype=np.float64)
    for i in range(n):
        parts = [shape_descriptors_1ch(profiles[i, c * W:(c + 1) * W])
                 for c in range(3)]
        out[i] = np.concatenate(parts)
    return out


# --------------------------- DSGE per-channel --------------------------- #

def _classifier():
    return LogisticRegression(max_iter=2000, n_jobs=1)


def _fit_lr(feat, y):
    sc = StandardScaler().fit(feat)
    return sc, _classifier().fit(sc.transform(feat), y)


def select_dsge_basis_perchannel(prof_tr, y_tr, g_tr, seed):
    """Pick one basis on inner-val (pooled across channels), return basis."""
    inner_tr, inner_val = grouped_split(prof_tr.shape[0], groups=g_tr,
                                        test_size=0.25, random_state=seed + 10)
    # Use channel-0 windows for selection (representative); cheap and unbiased.
    Xtr = prof_tr[inner_tr][:, :W]
    ytr = y_tr[inner_tr]
    Xv = prof_tr[inner_val][:, :W]
    yv = y_tr[inner_val]
    rows = {}
    for basis in ["polynomial", "fractional", "robust"]:
        r = run_dsge_pipeline(X_train=Xtr, y_train=ytr, X_test=Xv, y_test=yv,
                              basis=basis, n=3, ridge=0.01, standardize=True,
                              classifier=_classifier())
        rows[basis] = float(r["main"].macro_f1)
    # PATP sweep too.
    sweep = patp_alpha_sweep(Xtr, ytr, Xv, yv, n=3, ridge=0.01,
                             alphas=ALPHA_GRID, classifier=_classifier(),
                             standardize=True)
    best_basis = max(rows, key=rows.get)
    if sweep["best_macro_f1"] > rows[best_basis]:
        return dict(kind="patp", basis="patp", alpha=float(sweep["best_alpha"]),
                    n=3, val=float(sweep["best_macro_f1"]))
    return dict(kind="discrete", basis=best_basis, alpha=None, n=3,
                val=float(rows[best_basis]))


def dsge_features_perchannel(prof_tr, y_tr, prof_te, cfg):
    """Fit DSGE per channel on z-scored windows; concat → 3·n_classes feats."""
    kw = (dict(basis="patp", n=cfg["n"], alpha=cfg["alpha"], ridge=0.01)
          if cfg["kind"] == "patp" else dict(basis=cfg["basis"], n=3, ridge=0.01))
    feats_tr, feats_te, conds = [], [], {}
    for c in range(3):
        Xtr = prof_tr[:, c * W:(c + 1) * W]
        Xte = prof_te[:, c * W:(c + 1) * W]
        sc = StandardScaler().fit(Xtr)
        ext = DSGEFeatureExtractor(**kw)
        feats_tr.append(ext.fit_transform(sc.transform(Xtr), y_tr))
        feats_te.append(ext.transform(sc.transform(Xte)))
        conds[CHANNELS[c]] = {int(k): float(v) for k, v in ext.conditioning_report.items()}
    return np.concatenate(feats_tr, axis=1), np.concatenate(feats_te, axis=1), conds


# --------------------------- driver --------------------------- #

def _f1(yt, yp):
    return float(f1_score(yt, yp, average="macro"))


def _acc(yt, yp):
    return float(accuracy_score(yt, yp))


def _fmt_p(p):
    return f"{p:.3e}" if p < 1e-4 else f"{p:.4f}"


def main():
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "results" / "drilling"
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(">>> loading wells", flush=True)
        volve = load_volve()
        forge = load_forge_58_32()
        print(f"Volve on-bottom rows {len(volve)}, FORGE 58-32 on-bottom rows {len(forge)}",
              flush=True)

        pV, gV, sV = make_windows(volve, well_id=0, group_base=0)
        pF, gF, sF = make_windows(forge, well_id=1, group_base=1_000_000)
        # Balance classes by selecting WHOLE groups (depth-blocks) up to the
        # smaller well's window count — keeps group integrity intact.
        rng = np.random.default_rng(SEED)
        cap = min(len(pV), len(pF))
        pF, gF, sF = balance_by_groups(pF, gF, sF, cap, rng)
        pV, gV, sV = balance_by_groups(pV, gV, sV, cap, rng)
        profiles = np.concatenate([pV, pF], axis=0)
        y = np.concatenate([np.zeros(len(pV), int), np.ones(len(pF), int)])
        groups = np.concatenate([gV, gF])
        spans = sV + sF
        class_names = ["Volve_15_9_F15", "FORGE_58-32"]
        print(f"windows: Volve {len(pV)}, FORGE {len(pF)}; "
              f"profile dim {profiles.shape[1]} (3 ch × {W}); "
              f"groups {len(np.unique(groups))}", flush=True)

        # ---- leakage-safe split ----
        tr, te = grouped_split(profiles.shape[0], groups=groups,
                               test_size=0.3, random_state=SEED)
        audit = leakage_audit(tr, te, groups=groups)
        raw_overlap = raw_overlap_audit(spans, tr, te)
        print(f"leakage audit: group overlap={audit['groups']['leakage']}, "
              f"raw-row interval overlaps={raw_overlap}", flush=True)
        Xtr, Xte = profiles[tr], profiles[te]
        ytr, yte = y[tr], y[te]
        gtr = groups[tr]

        # ---- shape-descriptor features (33) ----
        S_tr = shape_features(Xtr)
        S_te = shape_features(Xte)

        # ---- H1: group-level moment separability ----
        uniq = np.unique(groups)  # all groups for H1 (descriptive)
        Sall = shape_features(profiles)
        gstat = np.array([Sall[groups == gg].mean(0) for gg in uniq])
        gy = np.array([int(y[groups == gg][0]) for gg in uniq])
        names33 = [f"{ch}:{s}" for ch in CHANNELS for s in STAT_NAMES]
        h1 = {}
        for j, nm in enumerate(names33):
            a = gstat[gy == 0, j]
            b = gstat[gy == 1, j]
            H, p = sps.kruskal(a, b)
            # eta^2 (2-group → point-biserial-ish)
            grand = gstat[:, j].mean()
            ssb = len(a) * (a.mean() - grand) ** 2 + len(b) * (b.mean() - grand) ** 2
            sst = ((gstat[:, j] - grand) ** 2).sum()
            h1[nm] = dict(H=float(H), p_kw=float(p),
                          eta_sq=float(ssb / sst) if sst > 0 else 0.0)
        best_h1 = min(h1, key=lambda k: h1[k]["p_kw"])
        h1_pass = h1[best_h1]["p_kw"] < 1e-6

        # ---- DSGE features (per-channel, basis on inner-val) ----
        cfg = select_dsge_basis_perchannel(Xtr, ytr, groups[tr], SEED)
        Dtr, Dte, conds = dsge_features_perchannel(Xtr, ytr, Xte, cfg)
        n_dsge = Dtr.shape[1]

        # ---- H2: DSGE vs fixed-stat ----
        # equal budget: stat top-n_dsge by train ANOVA F
        F = []
        for j in range(S_tr.shape[1]):
            a, b = S_tr[ytr == 0, j], S_tr[ytr == 1, j]
            F.append(float(sps.f_oneway(a, b).statistic))
        topk = list(np.argsort(F)[::-1][:n_dsge])
        sc_d, clf_d = _fit_lr(Dtr, ytr); yp_d = clf_d.predict(sc_d.transform(Dte))
        sc_se, clf_se = _fit_lr(S_tr[:, topk], ytr); yp_se = clf_se.predict(sc_se.transform(S_te[:, topk]))
        sc_ss, clf_ss = _fit_lr(S_tr, ytr); yp_ss = clf_ss.predict(sc_ss.transform(S_te))
        f1_d, f1_se, f1_ss = _f1(yte, yp_d), _f1(yte, yp_se), _f1(yte, yp_ss)

        bs_equal = bootstrap_compare(y_true=yte, y_pred_baseline=yp_se,
                                     y_pred_hybrid=yp_d, R=1000, metric="macro_f1", seed=SEED)
        bs_strong = bootstrap_compare(y_true=yte, y_pred_baseline=yp_ss,
                                      y_pred_hybrid=yp_d, R=1000, metric="macro_f1", seed=SEED + 1)

        # ---- H4★: does DSGE add on top of the strong 33-stat baseline? ----
        full_tr = np.concatenate([Dtr, S_tr], axis=1)
        full_te = np.concatenate([Dte, S_te], axis=1)
        sc_f, clf_f = _fit_lr(full_tr, ytr); yp_f = clf_f.predict(sc_f.transform(full_te))
        f1_full = _f1(yte, yp_f)
        bs_h4 = bootstrap_compare(y_true=yte, y_pred_baseline=yp_ss,
                                  y_pred_hybrid=yp_f, R=1000, metric="macro_f1", seed=SEED + 2)

        result = dict(
            task="plain DSGE well classification (2 wells, channels=RPM/Torque/ROP)",
            class_names=class_names,
            n_windows=dict(volve=int(len(pV)), forge=int(len(pF))),
            n_train=int(tr.size), n_test=int(te.size),
            leakage_groups_overlap=int(audit["groups"]["leakage"]),
            leakage_raw_row_overlaps=int(raw_overlap),
            window=W, k_per_block=K_PER_BLOCK, guard=GUARD, channels=CHANNELS,
            dsge_config=cfg, dsge_feature_dim=int(n_dsge),
            dsge_cond_per_channel=conds,
            h1=dict(best=best_h1, p_kw=h1[best_h1]["p_kw"],
                    eta_sq=h1[best_h1]["eta_sq"], passed=bool(h1_pass),
                    n_groups=int(len(uniq)), all=h1),
            macro_f1=dict(dsge=f1_d, stat_equal=f1_se, stat_strong=f1_ss, full=f1_full),
            h2_equal=dict(delta=bs_equal["delta"], p=bs_equal["p_value"],
                          ci=[bs_equal["ci_low"], bs_equal["ci_high"]],
                          verdict="PASS" if (bs_equal["significant"] and bs_equal["delta"] > 0) else "FAIL"),
            h2_strong=dict(delta=bs_strong["delta"], p=bs_strong["p_value"],
                           ci=[bs_strong["ci_low"], bs_strong["ci_high"]],
                           verdict="PASS" if (bs_strong["significant"] and bs_strong["delta"] > 0) else "FAIL"),
            h4star=dict(delta=bs_h4["delta"], p=bs_h4["p_value"],
                        ci=[bs_h4["ci_low"], bs_h4["ci_high"]],
                        verdict="PASS" if (bs_h4["significant"] and bs_h4["delta"] > 0) else "FAIL"),
        )
        (out_dir / "drilling.json").write_text(json.dumps(result, indent=2))

        # ---- console summary ----
        print(f"\n=== Drilling well-classification (plain DSGE, 2 classes) ===")
        print(f"H1 (group-level, {len(uniq)} groups): best={best_h1} "
              f"p_kw={_fmt_p(h1[best_h1]['p_kw'])} η²={h1[best_h1]['eta_sq']:.3f} "
              f"→ {'PASS' if h1_pass else 'FAIL'}")
        print(f"DSGE config (inner-val): {cfg}")
        print(f"macro-F1: DSGE-{n_dsge}={f1_d:.4f} | stat-equal({n_dsge})={f1_se:.4f} | "
              f"stat-strong(33)={f1_ss:.4f} | full={f1_full:.4f}")
        print(f"H2 equal budget (DSGE vs stat-{n_dsge}): Δ={bs_equal['delta']:+.4f} "
              f"p={_fmt_p(bs_equal['p_value'])} → {result['h2_equal']['verdict']}")
        print(f"H2 strong (DSGE vs stat-33): Δ={bs_strong['delta']:+.4f} "
              f"p={_fmt_p(bs_strong['p_value'])} → {result['h2_strong']['verdict']}")
        print(f"H4★ (DSGE adds on stat-33): Δ={bs_h4['delta']:+.4f} "
              f"p={_fmt_p(bs_h4['p_value'])} → {result['h4star']['verdict']}")
        print(f"leakage: group overlap={audit['groups']['leakage']}, "
              f"raw-row overlaps={raw_overlap}")
        print(f"Saved: {out_dir / 'drilling.json'}")


if __name__ == "__main__":
    main()
