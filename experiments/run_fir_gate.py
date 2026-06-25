"""FIR-robustness gate (Paper 2, thesis-critical) — does the cepstral
representation confer robustness to convolutional (multipath) distortion that
the spectral representation lacks?

Homomorphic-deconvolution thesis: a convolutional channel `y = x*h` adds the
smooth term `log|H|` to the log-spectrum, which lives at **low quefrency** in
the cepstrum. So a low-quefrency-liftered cepstrum is approximately
channel-invariant, whereas the spectral shape descriptors are tilted by
`log|H|`. We test this directly with the C2 classifier on three
representations, training on CLEAN signals and testing under an FIR-channel
sweep on held-out (leakage-safe) windows.

Representations (each → 11 descriptors → C2):
  - spectral        : log-magnitude STFT profile → spectral_statistics_extended
  - cepstral-full   : real cepstrum (drop only c[0]) → shape descriptors
  - cepstral-lifter : real cepstrum dropping low quefrency (channel band) → "

Pass criterion (FIXED BEFORE seeing numbers, mirroring the H3 rule):
  retention_L≥3 = mean macro-F1 over FIR lengths L≥3 / clean (L=1) macro-F1.
  PASS  iff  retention(cepstral-lifter) ≥ 0.85
        AND  retention(cepstral-lifter) − retention(spectral) ≥ 0.10.
A pass → Paper 2 is a positive invariance study; a fail → honest limits note.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_TOOLKIT = Path(__file__).resolve().parents[1] / "toolkit"
if str(_TOOLKIT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT))
from splits import grouped_split  # noqa: E402
from stats_basis import c2_kunchenko_mahalanobis  # noqa: E402

from real_data import discover_recordings, read_iq_window, group_id_for  # noqa: E402
from features import stft_log_profile, stft_cepstrum  # noqa: E402
from h2_equal_budget import spectral_statistics_extended  # noqa: E402
from run_drilling import shape_descriptors_1ch, STAT_NAMES  # noqa: E402
from synthetic import make_fir_channel, apply_fir_channel  # noqa: E402

SEED = 2026
WIN = 4096
N_FFT = 256
HOP = 128
K_PER_BLOCK = 4
GUARD = WIN
N_BLOCKS = 20
MAX_WIN_PER_CLASS = 120
LIFT_LO = 8           # drop the lowest 8 quefrencies (channel/envelope band)
FIR_SWEEP = [1, 2, 3, 5, 8, 12]   # L=1 ⇒ clean (identity)
ENERGY_STD_THRESH = 0.8


def _lr():
    return LogisticRegression(max_iter=2000, n_jobs=1)


def _f1(yt, yp):
    return float(f1_score(yt, yp, average="macro"))


# ---------------- descriptor extractors (window → per-frame 11-d S) -------- #

def spectral_S(iq):
    prof = stft_log_profile(iq, fs=120e6, n_fft=N_FFT, hop=HOP)
    mask = prof.std(axis=1) >= ENERGY_STD_THRESH
    if not np.any(mask):
        return None
    return spectral_statistics_extended(prof[mask])


def _cep_S(iq, lifter_lo):
    cep = stft_cepstrum(iq, fs=120e6, n_fft=N_FFT, hop=HOP, lifter_lo=lifter_lo)
    # energy filter on the parallel log-profile (same frames kept as spectral)
    prof = stft_log_profile(iq, fs=120e6, n_fft=N_FFT, hop=HOP)
    mask = prof.std(axis=1) >= ENERGY_STD_THRESH
    if not np.any(mask):
        return None
    cep = cep[mask]
    out = np.empty((cep.shape[0], len(STAT_NAMES)), dtype=np.float64)
    for i in range(cep.shape[0]):
        v = cep[i]
        s = v.std()
        vz = (v - v.mean()) / s if s > 1e-9 else v - v.mean()
        out[i] = shape_descriptors_1ch(vz)
    return out


def cepstral_full_S(iq):
    return _cep_S(iq, lifter_lo=1)


def cepstral_lift_S(iq):
    return _cep_S(iq, lifter_lo=LIFT_LO)


REPRS = {"spectral": spectral_S,
         "cepstral_full": cepstral_full_S,
         "cepstral_lifter": cepstral_lift_S}


# ---------------- raw-window dataset (leakage-safe blocks) ---------------- #

def build_raw_windows():
    recs = discover_recordings("2G")
    rng = np.random.default_rng(SEED)
    label_ids = {}
    for r in recs:
        label_ids.setdefault(r.class_label, len(label_ids))
    windows, ys, gs = [], [], []
    per_class = {c: 0 for c in label_ids.values()}
    for rec_idx, rec in enumerate(recs):
        cid = label_ids[rec.class_label]
        if per_class[cid] >= MAX_WIN_PER_CLASS:
            continue
        block_size = rec.n_complex // N_BLOCKS
        pos = 0
        blk = 0
        while pos + K_PER_BLOCK * WIN <= rec.n_complex and per_class[cid] < MAX_WIN_PER_CLASS:
            gid = group_id_for(rec_idx, blk, N_BLOCKS)
            for k in range(K_PER_BLOCK):
                if per_class[cid] >= MAX_WIN_PER_CLASS:
                    break
                s = pos + k * WIN
                iq = read_iq_window(rec.path, start_sample=s, sample_count=WIN)
                windows.append(iq); ys.append(cid); gs.append(gid)
                per_class[cid] += 1
            pos += K_PER_BLOCK * WIN + GUARD
            blk += 1
    return windows, np.array(ys), np.array(gs), list(label_ids)


def frames_from(windows, idx, y, extractor, fir_taps=None):
    """Apply optional FIR, extract per-frame descriptors, expand labels."""
    S_list, y_list = [], []
    for i in idx:
        iq = windows[i] if fir_taps is None else apply_fir_channel(windows[i], fir_taps[i])
        S = extractor(iq)
        if S is None:
            continue
        S_list.append(S); y_list.append(np.full(S.shape[0], y[i], np.int64))
    return np.concatenate(S_list), np.concatenate(y_list)


def main():
    out_dir = _HERE.parent / "results" / "fir_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(">>> building raw RF windows", flush=True)
        windows, y, g, names = build_raw_windows()
        n = len(windows)
        print(f"windows={n}, classes={len(names)}, groups={len(np.unique(g))}", flush=True)
        tr, te = grouped_split(n, groups=g, test_size=0.3, random_state=SEED)

        # Pre-draw a fixed random FIR channel per (test window, severity) for
        # reproducibility; L=1 is identity (clean).
        rng = np.random.default_rng(SEED + 7)
        chan = {L: {i: make_fir_channel(L, rng) for i in te} for L in FIR_SWEEP}

        results = {}
        for rep, extractor in REPRS.items():
            # train on CLEAN train windows
            S_tr, y_tr = frames_from(windows, tr, y, extractor, fir_taps=None)
            sweep = []
            for L in FIR_SWEEP:
                taps = None if L == 1 else chan[L]
                S_te, y_te = frames_from(windows, te, y, extractor, fir_taps=taps)
                c2_tr, c2_te = c2_kunchenko_mahalanobis(S_tr, y_tr, S_te)
                sc = StandardScaler().fit(c2_tr)
                clf = _lr().fit(sc.transform(c2_tr), y_tr)
                f1 = _f1(y_te, clf.predict(sc.transform(c2_te)))
                sweep.append(dict(L=L, macro_f1=f1))
                print(f"  {rep:<16} L={L:>2}: macro_f1={f1:.4f}", flush=True)
            clean = sweep[0]["macro_f1"]
            ret = float(np.mean([r["macro_f1"] for r in sweep if r["L"] >= 3]) / max(clean, 1e-9))
            results[rep] = dict(sweep=sweep, clean_f1=clean, retention_Lge3=ret)

        ret_lift = results["cepstral_lifter"]["retention_Lge3"]
        ret_spec = results["spectral"]["retention_Lge3"]
        passed = bool(ret_lift >= 0.85 and (ret_lift - ret_spec) >= 0.10)
        verdict = dict(
            criterion="retention(cepstral_lifter)>=0.85 AND minus retention(spectral)>=0.10",
            retention_cepstral_lifter=ret_lift,
            retention_cepstral_full=results["cepstral_full"]["retention_Lge3"],
            retention_spectral=ret_spec,
            gap_lifter_minus_spectral=float(ret_lift - ret_spec),
            passed=passed,
            verdict="PASS — Paper 2 positive invariance study"
                    if passed else "FAIL — Paper 2 downgrades to limits note",
        )
        out = dict(config=dict(seed=SEED, win=WIN, n_fft=N_FFT, hop=HOP,
                               lifter_lo=LIFT_LO, fir_sweep=FIR_SWEEP,
                               n_windows=n, classes=names),
                   per_representation=results, gate=verdict)
        (out_dir / "fir_gate.json").write_text(json.dumps(out, indent=2))

        print("\n=== FIR-robustness gate ===")
        for rep in REPRS:
            r = results[rep]
            print(f"  {rep:<16} clean={r['clean_f1']:.4f}  retention(L>=3)={r['retention_Lge3']:.3f}")
        print(f"\ncepstral_lifter retention={ret_lift:.3f}, spectral retention={ret_spec:.3f}, "
              f"gap={ret_lift-ret_spec:+.3f}")
        print(f"GATE: {verdict['verdict']}")
        print(f"Saved {out_dir / 'fir_gate.json'}")


if __name__ == "__main__":
    main()
