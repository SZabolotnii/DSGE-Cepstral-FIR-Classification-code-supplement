"""Revision E8 — DroneRF repeated-split FIR-robustness CV (cross-domain hardening).

Reviewer (DSP, 2026) Q5 / external validity: the RF evidence was a single split.
This upgrades it to the SAME repeated-split + bootstrap protocol used on CWRU,
on the locally-available DroneRF 2G data, with the SAME pre-registered lifter
l=3 and the SAME identical-descriptor pipeline as the CWRU headline. Channels
remain synthetically injected FIR (complex multipath); measured-channel
validation stays an explicit limitation.

PRE-REGISTERED criterion (identical to CWRU, fixed before running):
  At lifter l=3, cepstral-DBMSED mean macro-F1 UNDER FIR (L>=3) exceeds
  spectral-DBMSED under FIR, with the per-split gap (i) positive under a
  one-sided Wilcoxon test p<0.05 AND (ii) a 95% bootstrap CI (R=2000) on its
  mean excluding zero, across N_SPLITS=15 leakage-safe repeated splits.

Discipline: leakage-safe (recording, time-block) groups; equal 11-feature
budget; identical descriptors both branches; repeated-split CV + bootstrap.
Energy masking drops noise-floor frames (DroneRF captures the full 120 MHz band
but each emitter is narrowband), identically for both representations.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats as sps

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_TOOLKIT = Path(__file__).resolve().parents[1] / "toolkit"
if str(_TOOLKIT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT))

from run_fir_gate import build_raw_windows, ENERGY_STD_THRESH, N_FFT, HOP, WIN  # noqa: E402
from run_fir_cwru_cv import (                                    # noqa: E402
    _cep_descriptors, _c2_lr_f1, _boot_ci, _under, FIR_SWEEP, PREREG_LIFT,
    N_SPLITS, TEST_SIZE, SEED, DESC_MODE,
)
from features import stft_log_profile, stft_cepstrum            # noqa: E402
from synthetic import make_fir_channel, apply_fir_channel       # noqa: E402
from splits import grouped_split                                # noqa: E402

FS_RF = 120e6


def _profile_mask(iq):
    prof = stft_log_profile(iq, fs=FS_RF, n_fft=N_FFT, hop=HOP)
    return prof, prof.std(axis=1) >= ENERGY_STD_THRESH


def spectral_S(iq):
    prof, m = _profile_mask(iq)
    if not np.any(m):
        return None
    return _cep_descriptors(prof[m])           # identical descriptors (E0 pipeline)


def cep_S(iq):
    prof, m = _profile_mask(iq)
    if not np.any(m):
        return None
    cep = stft_cepstrum(iq, fs=FS_RF, n_fft=N_FFT, hop=HOP, lifter_lo=PREREG_LIFT)
    return _cep_descriptors(cep[m])


def _frames(feat_fn, windows, idx, y, fir_taps):
    Ss, ys = [], []
    for i in idx:
        iq = windows[i] if fir_taps is None else apply_fir_channel(windows[i], fir_taps[i])
        S = feat_fn(iq)
        if S is None:
            continue
        Ss.append(S)
        ys.append(np.full(S.shape[0], y[i], np.int64))
    return np.concatenate(Ss), np.concatenate(ys)


def _method_under(feat_fn, windows, tr, te, y, chan):
    S_tr, y_tr = _frames(feat_fn, windows, tr, y, None)
    sweep = {}
    for L in FIR_SWEEP:
        S_te, y_te = _frames(feat_fn, windows, te, y, None if L == 1 else chan[L])
        sweep[L] = _c2_lr_f1(S_tr, y_tr, S_te, y_te)
    return dict(clean=sweep[1], under_fir=_under(sweep),
                sweep={int(L): sweep[L] for L in FIR_SWEEP})


def eval_one_split(windows, y, g, rs: int) -> dict:
    tr, te = grouped_split(len(windows), groups=g, test_size=TEST_SIZE, random_state=rs)
    rng = np.random.default_rng(rs + 7)
    chan = {L: {int(i): make_fir_channel(L, rng) for i in te} for L in FIR_SWEEP if L > 1}
    spec = _method_under(spectral_S, windows, tr, te, y, chan)
    cep = _method_under(cep_S, windows, tr, te, y, chan)
    return dict(rs=rs, spectral=spec, cepstral=cep,
                gap=float(cep["under_fir"] - spec["under_fir"]))


def main():
    out_dir = _HERE.parent / "results" / "fir_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(f">>> building DroneRF 2G windows (desc_mode={DESC_MODE})", flush=True)
        windows, y, g, names = build_raw_windows()
        print(f"windows={len(windows)} classes={len(names)} groups={len(np.unique(g))}",
              flush=True)

        per_split = []
        for i in range(N_SPLITS):
            r = eval_one_split(windows, y, g, rs=SEED + i)
            per_split.append(r)
            print(f"  split {i:2d}: spec_underFIR={r['spectral']['under_fir']:.4f} "
                  f"cep_underFIR={r['cepstral']['under_fir']:.4f} gap={r['gap']:+.4f}",
                  flush=True)

        gaps = np.array([r["gap"] for r in per_split])
        spec_under = np.array([r["spectral"]["under_fir"] for r in per_split])
        cep_under = np.array([r["cepstral"]["under_fir"] for r in per_split])
        spec_clean = np.array([r["spectral"]["clean"] for r in per_split])
        cep_clean = np.array([r["cepstral"]["clean"] for r in per_split])

        w = sps.wilcoxon(cep_under, spec_under, alternative="greater")
        rng = np.random.default_rng(SEED + 99)
        ci_lo, ci_hi = _boot_ci(gaps, rng)
        passed = bool((w.pvalue < 0.05) and (ci_lo > 0))

        verdict = dict(
            pre_registered_criterion=(
                "lifter l=3: cepstral-DBMSED mean macro-F1 under FIR (L>=3) > spectral, "
                "one-sided Wilcoxon p<0.05 AND 95% bootstrap CI on per-split gap excluding "
                "zero, over 15 leakage-safe splits (DroneRF 2G; injected complex FIR)."),
            prereg_lift=PREREG_LIFT, n_splits=N_SPLITS,
            gap_mean=float(gaps.mean()), gap_std=float(gaps.std()),
            gap_min=float(gaps.min()), gap_max=float(gaps.max()),
            frac_wins=float((gaps > 0).mean()),
            wilcoxon_p=float(w.pvalue), bootstrap_ci95=[ci_lo, ci_hi], passed=passed,
            verdict=("PASS — the cepstral FIR-robustness advantage holds on a second "
                     "modality (RF) under repeated-split + bootstrap (injected channels)")
                    if passed else
                    ("FAIL/INCONCLUSIVE — RF repeated-split does not confirm under the "
                     "pre-registered test; report as honest cross-domain limit"),
        )
        out = dict(
            config=dict(dataset="DroneRF 2G", seed=SEED, fs=FS_RF, win=WIN, n_fft=N_FFT,
                        hop=HOP, prereg_lift=PREREG_LIFT, fir_sweep=FIR_SWEEP,
                        n_splits=N_SPLITS, test_size=TEST_SIZE, desc_mode=DESC_MODE,
                        n_windows=len(windows), n_groups=int(len(np.unique(g))),
                        classes=names, channel="injected complex FIR (synthetic.make_fir_channel)",
                        equal_feature_budget="11 identical z-norm shape descriptors -> DB-MSED"),
            aggregate=dict(
                spectral=dict(clean_mean=float(spec_clean.mean()), clean_std=float(spec_clean.std()),
                              under_fir_mean=float(spec_under.mean()), under_fir_std=float(spec_under.std())),
                cepstral=dict(clean_mean=float(cep_clean.mean()), clean_std=float(cep_clean.std()),
                              under_fir_mean=float(cep_under.mean()), under_fir_std=float(cep_under.std()))),
            verdict=verdict, per_split=per_split,
        )
        (out_dir / "fir_rf_cv.json").write_text(json.dumps(out, indent=2))

        print(f"\n=== DroneRF 2G FIR-robustness repeated-split CV (desc_mode={DESC_MODE}) ===")
        print(f"  spectral  clean={spec_clean.mean():.4f}  under_FIR={spec_under.mean():.4f}")
        print(f"  cepstral  clean={cep_clean.mean():.4f}  under_FIR={cep_under.mean():.4f}")
        print(f"  gap mean={gaps.mean():+.4f}  wins={int((gaps>0).sum())}/{N_SPLITS}  "
              f"Wilcoxon p={w.pvalue:.2e}  boot95=[{ci_lo:+.4f},{ci_hi:+.4f}]")
        print(f"  VERDICT: {verdict['verdict']}")
        print(f"  Saved {out_dir / 'fir_rf_cv.json'}")


if __name__ == "__main__":
    main()
