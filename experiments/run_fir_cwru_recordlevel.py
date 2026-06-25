"""Revision E7 — recording-level aggregation of frame decisions.

Reviewer (DSP, 2026) Q7: "How are frame-level descriptors aggregated to
recording-level decisions, or are windows treated independently? If
independent, have you evaluated simple voting/averaging to stabilize per-record
outcomes?"

The headline reports per-frame macro-F1. Here we aggregate the per-frame
DB-MSED decisions of each test window (one 4096-sample window = one short
recording snippet, ~31 frames) to a single per-window decision by (a) majority
vote and (b) mean class-probability, and report window-level macro-F1 alongside
frame-level, for spectral and cepstral (l=3) at matched 11-budget.

PRE-REGISTERED: the cepstral-over-spectral under-FIR advantage persists at
window level (one-sided Wilcoxon p<0.05 and bootstrap CI excluding zero on the
window-level mean-probability gap), removing the per-frame pseudoreplication
concern.

Discipline: identical (z-norm shape) descriptors both branches; FIR
channels/splits identical to the headline; 15 leakage-safe splits.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_TOOLKIT = Path(__file__).resolve().parents[1] / "toolkit"
if str(_TOOLKIT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT))

from run_fir_cwru_cv import (                                    # noqa: E402
    build_cwru_windows, make_real_fir, apply_real_fir, _boot_ci, _cep_descriptors,
    spectral_S, _under, FS, N_FFT, HOP, FIR_SWEEP, PREREG_LIFT, N_SPLITS,
    TEST_SIZE, SEED, DESC_MODE,
)
from features import stft_cepstrum                              # noqa: E402
from stats_basis import c2_kunchenko_mahalanobis               # noqa: E402
from splits import grouped_split                                # noqa: E402


def cep_l3_S(x):
    return _cep_descriptors(stft_cepstrum(x, fs=FS, n_fft=N_FFT, hop=HOP, lifter_lo=PREREG_LIFT))


def _frames(feat_fn, windows, idx, y, fir_taps):
    Ss, ys, wid = [], [], []
    for i in idx:
        xx = windows[i] if fir_taps is None else apply_real_fir(windows[i], fir_taps[i])
        S = feat_fn(xx)
        Ss.append(S)
        ys.append(np.full(S.shape[0], y[i], np.int64))
        wid.append(np.full(S.shape[0], i, np.int64))
    return np.concatenate(Ss), np.concatenate(ys), np.concatenate(wid)


def _three_level_f1(S_tr, y_tr, S_te, y_te, wid_te):
    """Frame-level, window-vote, and window-mean-proba macro-F1 from one fit."""
    c2_tr, c2_te = c2_kunchenko_mahalanobis(S_tr, y_tr, S_te)
    sc = StandardScaler().fit(c2_tr)
    clf = LogisticRegression(max_iter=2000, n_jobs=1).fit(sc.transform(c2_tr), y_tr)
    Xte = sc.transform(c2_te)
    yp = clf.predict(Xte)
    proba = clf.predict_proba(Xte)
    classes = clf.classes_
    frame_f1 = float(f1_score(y_te, yp, average="macro"))

    uw = np.unique(wid_te)
    y_true_w, y_vote_w, y_prob_w = [], [], []
    for w in uw:
        m = wid_te == w
        y_true_w.append(y_te[m][0])
        # majority vote
        vals, cnts = np.unique(yp[m], return_counts=True)
        y_vote_w.append(vals[cnts.argmax()])
        # mean class-probability
        y_prob_w.append(classes[proba[m].mean(0).argmax()])
    vote_f1 = float(f1_score(y_true_w, y_vote_w, average="macro"))
    prob_f1 = float(f1_score(y_true_w, y_prob_w, average="macro"))
    return frame_f1, vote_f1, prob_f1


def eval_one_split(windows, y, g, rs: int) -> dict:
    tr, te = grouped_split(len(windows), groups=g, test_size=TEST_SIZE, random_state=rs)
    rng = np.random.default_rng(rs + 7)
    chan = {L: {int(i): make_real_fir(L, rng) for i in te} for L in FIR_SWEEP if L > 1}

    res = {}
    for name, fn in (("spectral", spectral_S), ("cepstral", cep_l3_S)):
        S_tr, y_tr, _ = _frames(fn, windows, tr, y, None)
        sw = {lvl: {} for lvl in ("frame", "vote", "prob")}
        for L in FIR_SWEEP:
            S_te, y_te, wid_te = _frames(fn, windows, te, y, None if L == 1 else chan[L])
            f, v, p = _three_level_f1(S_tr, y_tr, S_te, y_te, wid_te)
            sw["frame"][L], sw["vote"][L], sw["prob"][L] = f, v, p
        res[name] = {lvl: dict(clean=sw[lvl][1], under_fir=_under(sw[lvl]))
                     for lvl in sw}
    return res


def main():
    out_dir = _HERE.parent / "results" / "fir_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(f">>> building CWRU raw windows (desc_mode={DESC_MODE})", flush=True)
        windows, y, g, names = build_cwru_windows()
        per_split = []
        for i in range(N_SPLITS):
            r = eval_one_split(windows, y, g, rs=SEED + i)
            per_split.append(r)
            print(f"  split {i:2d}: frame gap={r['cepstral']['frame']['under_fir']-r['spectral']['frame']['under_fir']:+.4f} "
                  f"prob gap={r['cepstral']['prob']['under_fir']-r['spectral']['prob']['under_fir']:+.4f}",
                  flush=True)

        agg = {}
        for name in ("spectral", "cepstral"):
            for lvl in ("frame", "vote", "prob"):
                cu = np.array([r[name][lvl]["under_fir"] for r in per_split])
                cc = np.array([r[name][lvl]["clean"] for r in per_split])
                agg[f"{name}_{lvl}"] = dict(clean_mean=float(cc.mean()),
                                            under_fir_mean=float(cu.mean()))

        # pre-registered: window-level (mean-proba) gap significance
        gap = np.array([r["cepstral"]["prob"]["under_fir"] - r["spectral"]["prob"]["under_fir"]
                        for r in per_split])
        w = sps.wilcoxon(gap, alternative="greater")
        rng = np.random.default_rng(SEED + 99)
        ci_lo, ci_hi = _boot_ci(gap, rng)
        passed = bool((w.pvalue < 0.05) and (ci_lo > 0))

        out = dict(
            config=dict(dataset="CWRU 12k-DE 10-class", seed=SEED, desc_mode=DESC_MODE,
                        n_fft=N_FFT, hop=HOP, lifter=PREREG_LIFT, fir_sweep=FIR_SWEEP,
                        n_splits=N_SPLITS, aggregation=["majority_vote", "mean_probability"],
                        record_unit="one 4096-sample window"),
            levels=agg,
            window_prob_gap=dict(gap_mean=float(gap.mean()), frac_wins=float((gap > 0).mean()),
                                 wilcoxon_p=float(w.pvalue), bootstrap_ci95=[ci_lo, ci_hi],
                                 passed=passed),
            per_split=per_split,
        )
        (out_dir / "fir_cwru_recordlevel.json").write_text(json.dumps(out, indent=2))

        print("\n=== CWRU frame vs recording-level (under FIR) ===")
        for lvl in ("frame", "vote", "prob"):
            s = agg[f"spectral_{lvl}"]["under_fir_mean"]
            c = agg[f"cepstral_{lvl}"]["under_fir_mean"]
            print(f"  {lvl:5s}: spectral={s:.4f}  cepstral={c:.4f}  gap={c-s:+.4f}")
        print(f"  window mean-proba gap: mean={gap.mean():+.4f}  wins={int((gap>0).sum())}/{N_SPLITS}  "
              f"Wilcoxon p={w.pvalue:.2e}  boot95=[{ci_lo:+.4f},{ci_hi:+.4f}]  PASS={passed}")
        print(f"  Saved {out_dir / 'fir_cwru_recordlevel.json'}")


if __name__ == "__main__":
    main()
