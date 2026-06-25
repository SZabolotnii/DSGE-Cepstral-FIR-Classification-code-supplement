"""Revision E5+E6 — robustness of the cepstral FIR-advantage to DSP and
regularization hyperparameters.

Reviewer (DSP, 2026):
  Q3: sensitivity to NFFT, window/hop, frame length, epsilon in the log.
  Q1: covariance regularization in Eq. (5) — lambda value, shrinkage, cond(F),
      and sensitivity of results to that choice.

Design: ONE-AT-A-TIME (OAT) sensitivity around the headline baseline
(n_fft=256, hop=128, hann, cep-eps=1e-8, ridge=1e-2). For each config we report
the mean under-FIR gap (cepstral l=3 - spectral) over N_SENS leakage-safe splits
and whether the advantage keeps its (positive) sign. This is a
robustness-of-conclusion check: hyperparameters are NEVER tuned on the test
metric; we only verify the sign is stable.

We also report cond(F_reg) per class (the regularized within-class descriptor
covariance) at the baseline, for several ridge values, to document Eq. (5).

Local STFT functions mirror features.py exactly so the shared module is left
untouched; the baseline config reproduces the headline pipeline.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import stft as scipy_stft

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_TOOLKIT = Path(__file__).resolve().parents[1] / "toolkit"
if str(_TOOLKIT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT))

from run_fir_cwru_cv import (                                    # noqa: E402
    build_cwru_windows, make_real_fir, apply_real_fir, _c2_lr_f1, _cep_descriptors,
    _under, FS, N_FFT, HOP, FIR_SWEEP, PREREG_LIFT, TEST_SIZE, SEED,
)
from splits import grouped_split                                # noqa: E402

N_SENS = 8                       # splits per config (sensitivity scan, not headline)
PROFILE_EPS = 1e-12              # features.stft_log_profile default

BASE = dict(n_fft=256, hop=128, window="hann", cep_eps=1e-8)
# OAT grid: each entry overrides one factor of BASE.
DSP_CONFIGS = [
    ("baseline", {}),
    ("nfft128", dict(n_fft=128, hop=64)),
    ("nfft512", dict(n_fft=512, hop=256)),
    ("hop_quarter", dict(hop=64)),
    ("window_hamming", dict(window="hamming")),
    ("cepeps_1e-6", dict(cep_eps=1e-6)),
    ("cepeps_1e-10", dict(cep_eps=1e-10)),
]
RIDGES = [1e-3, 1e-2, 1e-1]


def _stft(x, n_fft, hop, window):
    _, _, Z = scipy_stft(x, fs=FS, window=window, nperseg=n_fft,
                         noverlap=n_fft - hop, return_onesided=False,
                         boundary=None, padded=False)
    return Z


def profile_S(x, n_fft, hop, window, cep_eps):
    Z = _stft(x, n_fft, hop, window)[1:, :]            # drop DC
    log_p = np.log(np.abs(Z) ** 2 + PROFILE_EPS)
    log_p = log_p - log_p.mean(0, keepdims=True)
    return _cep_descriptors(log_p.T)


def cep_S(x, n_fft, hop, window, cep_eps):
    Z = _stft(x, n_fft, hop, window)
    cep = np.fft.ifft(np.log(np.abs(Z) + cep_eps), axis=0).real
    lo, hi = max(1, PREREG_LIFT), n_fft // 2
    return _cep_descriptors(cep[lo:hi, :].T)


def _frames(feat_fn, cfg, windows, idx, y, fir_taps):
    Ss, ys = [], []
    for i in idx:
        xx = windows[i] if fir_taps is None else apply_real_fir(windows[i], fir_taps[i])
        S = feat_fn(xx, **cfg)
        Ss.append(S)
        ys.append(np.full(S.shape[0], y[i], np.int64))
    return np.concatenate(Ss), np.concatenate(ys)


def _under_method(feat_fn, cfg, windows, tr, te, y, chan):
    S_tr, y_tr = _frames(feat_fn, cfg, windows, tr, y, None)
    sweep = {}
    for L in FIR_SWEEP:
        S_te, y_te = _frames(feat_fn, cfg, windows, te, y, None if L == 1 else chan[L])
        sweep[L] = _c2_lr_f1(S_tr, y_tr, S_te, y_te)
    return _under(sweep)


def _cond_per_class(feat_fn, cfg, windows, tr, y, ridge):
    """cond(F_reg = cov(Zc) + ridge*I) per class on standardized clean features."""
    from sklearn.preprocessing import StandardScaler
    S_tr, y_tr = _frames(feat_fn, cfg, windows, tr, y, None)
    Z = StandardScaler().fit_transform(S_tr)
    d = Z.shape[1]
    conds = []
    for c in np.unique(y_tr):
        F = np.cov(Z[y_tr == c], rowvar=False) + ridge * np.eye(d)
        conds.append(float(np.linalg.cond(F)))
    return conds


def main():
    out_dir = _HERE.parent / "results" / "fir_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(">>> building CWRU raw windows", flush=True)
        windows, y, g, names = build_cwru_windows()
        splits = []
        for i in range(N_SENS):
            rs = SEED + i
            tr, te = grouped_split(len(windows), groups=g, test_size=TEST_SIZE, random_state=rs)
            rng = np.random.default_rng(rs + 7)
            chan = {L: {int(j): make_real_fir(L, rng) for j in te} for L in FIR_SWEEP if L > 1}
            splits.append((tr, te, chan))

        # E5 — DSP OAT sensitivity
        dsp = {}
        for name, override in DSP_CONFIGS:
            cfg = {**BASE, **override}
            gaps = []
            for (tr, te, chan) in splits:
                cu = _under_method(cep_S, cfg, windows, tr, te, y, chan)
                su = _under_method(profile_S, cfg, windows, tr, te, y, chan)
                gaps.append(cu - su)
            gaps = np.array(gaps)
            dsp[name] = dict(config=cfg, gap_mean=float(gaps.mean()),
                             gap_std=float(gaps.std()), gap_min=float(gaps.min()),
                             sign_stable=bool((gaps > 0).all()))
            print(f"  DSP {name:16s} gap={gaps.mean():+.4f}±{gaps.std():.4f} "
                  f"min={gaps.min():+.4f} sign_stable={(gaps>0).all()}", flush=True)

        # E6 — ridge / cond(F) sensitivity (baseline DSP config)
        ridge_sens = {}
        for ridge in RIDGES:
            # gap with this ridge (override the head's default by monkey-temporarily?)
            # _c2_lr_f1 uses default ridge=1e-2 inside c2_kunchenko_mahalanobis; re-implement
            # the under-FIR with explicit ridge via stats_basis.
            from stats_basis import c2_kunchenko_mahalanobis
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import f1_score

            def c2_lr_ridge(S_tr, y_tr, S_te, y_te, _r=ridge):
                c2_tr, c2_te = c2_kunchenko_mahalanobis(S_tr, y_tr, S_te, ridge=_r)
                sc = StandardScaler().fit(c2_tr)
                clf = LogisticRegression(max_iter=2000, n_jobs=1).fit(sc.transform(c2_tr), y_tr)
                return float(f1_score(y_te, clf.predict(sc.transform(c2_te)), average="macro"))

            gaps, conds_all = [], []
            for (tr, te, chan) in splits:
                # cepstral l=3 and spectral under-FIR with explicit ridge
                Sc_tr, yc_tr = _frames(cep_S, BASE, windows, tr, y, None)
                Sp_tr, yp_tr = _frames(profile_S, BASE, windows, tr, y, None)
                csweep, ssweep = {}, {}
                for L in FIR_SWEEP:
                    taps = None if L == 1 else chan[L]
                    Sc_te, yc_te = _frames(cep_S, BASE, windows, te, y, taps)
                    Sp_te, yp_te = _frames(profile_S, BASE, windows, te, y, taps)
                    csweep[L] = c2_lr_ridge(Sc_tr, yc_tr, Sc_te, yc_te)
                    ssweep[L] = c2_lr_ridge(Sp_tr, yp_tr, Sp_te, yp_te)
                gaps.append(_under(csweep) - _under(ssweep))
                conds_all += _cond_per_class(cep_S, BASE, windows, tr, y, ridge)
            gaps = np.array(gaps)
            ridge_sens[f"{ridge:g}"] = dict(
                gap_mean=float(gaps.mean()), gap_std=float(gaps.std()),
                sign_stable=bool((gaps > 0).all()),
                cond_F_median=float(np.median(conds_all)),
                cond_F_max=float(np.max(conds_all)))
            print(f"  ridge {ridge:<6g} gap={gaps.mean():+.4f} sign_stable={(gaps>0).all()} "
                  f"cond(F) median={np.median(conds_all):.1f} max={np.max(conds_all):.1f}",
                  flush=True)

        out = dict(
            config=dict(dataset="CWRU 12k-DE 10-class", seed=SEED, n_sens_splits=N_SENS,
                        baseline=BASE, headline_ridge=1e-2, profile_eps=PROFILE_EPS,
                        note="OAT sensitivity; hyperparameters never tuned on test metric"),
            dsp_sensitivity=dsp, ridge_sensitivity=ridge_sens,
            all_signs_stable=bool(all(d["sign_stable"] for d in dsp.values())
                                  and all(r["sign_stable"] for r in ridge_sens.values())),
        )
        (out_dir / "fir_cwru_sensitivity.json").write_text(json.dumps(out, indent=2))
        print(f"\n  all_signs_stable={out['all_signs_stable']}")
        print(f"  Saved {out_dir / 'fir_cwru_sensitivity.json'}")


if __name__ == "__main__":
    main()
