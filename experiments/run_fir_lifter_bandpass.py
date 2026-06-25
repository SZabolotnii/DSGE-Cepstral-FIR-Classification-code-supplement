"""Revision E3 — bandpass and smooth-taper lifter sweep.

Reviewer (DSP, 2026) Q4: "Why choose only a lower-bound lifter? Would a bandpass
lifter (or smooth taper) that also suppresses very high quefrencies further
improve robustness without harming clean accuracy?"

We compute the full real cepstrum once per window and apply four lifter
families by quefrency indexing:
  - lb:        lower-bound (incumbent), keep [lo, 128)
  - bp:        hard bandpass, keep [lo, hi)
  - bp_tukey:  bandpass with a Tukey (raised-cosine) taper over [lo, hi)
  - juang:     Juang-1987 sinusoidal lifter w[q]=1+(Lc/2)sin(pi*q/Lc) over [lo, hi)

PRE-REGISTERED (fixed BEFORE running): the headline lifter stays the incumbent
lower-bound l=3 (pre-registered from RF, untouched). E3 asks whether ANY
bandpass/tapered lifter beats lb(l=3) on mean under-FIR macro-F1 (L>=3) WITHOUT
worsening clean macro-F1 by more than MARGIN=0.02, over 15 leakage-safe splits.
A winner is reported as a candidate improvement; absence of one is reported as
"the lower-bound lifter is already on the robustness plateau."

Discipline: leakage-safe groups; FIR channels/splits identical to the headline;
equal 11-feature budget; 15 repeated splits.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import stft as scipy_stft, windows as sp_windows

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_TOOLKIT = Path(__file__).resolve().parents[1] / "toolkit"
if str(_TOOLKIT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT))

from run_fir_cwru_cv import (                                    # noqa: E402
    build_cwru_windows, make_real_fir, apply_real_fir, _c2_lr_f1, _cep_descriptors,
    _under, FS, N_FFT, HOP, FIR_SWEEP, PREREG_LIFT, N_SPLITS, TEST_SIZE, SEED,
)
from splits import grouped_split                                # noqa: E402

MARGIN = 0.02
HALF = N_FFT // 2                                                # 128

# lifter specs: (name, family, lo, hi)
LIFTERS = [
    ("lb3", "lb", PREREG_LIFT, HALF),        # incumbent reference
    ("lb2", "lb", 2, HALF),
    ("lb6", "lb", 6, HALF),
    ("bp_3_32", "bp", 3, 32),
    ("bp_3_64", "bp", 3, 64),
    ("bp_3_96", "bp", 3, 96),
    ("tukey_3_64", "bp_tukey", 3, 64),
    ("tukey_3_96", "bp_tukey", 3, 96),
    ("juang_3_128", "juang", 3, HALF),
]


def _cep_full(x: np.ndarray) -> np.ndarray:
    """Full real cepstrum, quefrency 0..HALF-1, shape (n_frames, HALF)."""
    _, _, Z = scipy_stft(x, fs=FS, window="hann", nperseg=N_FFT,
                         noverlap=N_FFT - HOP, return_onesided=False,
                         boundary=None, padded=False)
    cep = np.fft.ifft(np.log(np.abs(Z) + 1e-8), axis=0).real     # (N_FFT, n_frames)
    return cep[:HALF, :].T                                       # (n_frames, HALF)


def _lifter_weights(family: str, lo: int, hi: int) -> np.ndarray:
    """Multiplicative weights over the retained band [lo, hi)."""
    n = hi - lo
    if family in ("lb", "bp"):
        return np.ones(n)
    if family == "bp_tukey":
        return sp_windows.tukey(n, alpha=0.5) if n > 1 else np.ones(n)
    if family == "juang":
        q = np.arange(1, n + 1, dtype=np.float64)
        return 1.0 + (n / 2.0) * np.sin(np.pi * q / n)
    raise ValueError(family)


def lifter_S(cep_full: np.ndarray, family: str, lo: int, hi: int) -> np.ndarray:
    band = cep_full[:, lo:hi] * _lifter_weights(family, lo, hi)[None, :]
    return _cep_descriptors(band)


def _ceps_for(windows, idx, y, fir_taps):
    """Full cepstra + labels for a window set (one cepstrum pass each)."""
    ceps, ys = [], []
    for i in idx:
        xx = windows[i] if fir_taps is None else apply_real_fir(windows[i], fir_taps[i])
        ceps.append(_cep_full(xx))
        ys.append(i)
    return ceps, ys


def _stack(ceps, idx_ids, y, family, lo, hi):
    Ss, ys = [], []
    for cep, i in zip(ceps, idx_ids):
        S = lifter_S(cep, family, lo, hi)
        Ss.append(S)
        ys.append(np.full(S.shape[0], y[i], np.int64))
    return np.concatenate(Ss), np.concatenate(ys)


def eval_one_split(windows, y, g, rs: int) -> dict:
    tr, te = grouped_split(len(windows), groups=g, test_size=TEST_SIZE, random_state=rs)
    rng = np.random.default_rng(rs + 7)
    chan = {L: {int(i): make_real_fir(L, rng) for i in te} for L in FIR_SWEEP if L > 1}

    cep_tr, id_tr = _ceps_for(windows, tr, y, None)
    cep_te = {L: _ceps_for(windows, te, y, None if L == 1 else chan[L]) for L in FIR_SWEEP}

    out = {}
    for name, fam, lo, hi in LIFTERS:
        S_tr, y_tr = _stack(cep_tr, id_tr, y, fam, lo, hi)
        sweep = {}
        for L in FIR_SWEEP:
            ceps, ids = cep_te[L]
            S_te, y_te = _stack(ceps, ids, y, fam, lo, hi)
            sweep[L] = _c2_lr_f1(S_tr, y_tr, S_te, y_te)
        out[name] = dict(clean=sweep[1], under_fir=_under(sweep))
    return out


def main():
    out_dir = _HERE.parent / "results" / "fir_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(">>> building CWRU raw windows", flush=True)
        windows, y, g, names = build_cwru_windows()
        per_split = []
        for i in range(N_SPLITS):
            r = eval_one_split(windows, y, g, rs=SEED + i)
            per_split.append(r)
            print(f"  split {i:2d}: lb3 under_FIR={r['lb3']['under_fir']:.4f}", flush=True)

        agg = {}
        for name, *_ in LIFTERS:
            cu = np.array([r[name]["under_fir"] for r in per_split])
            cc = np.array([r[name]["clean"] for r in per_split])
            agg[name] = dict(clean_mean=float(cc.mean()), under_fir_mean=float(cu.mean()),
                             under_fir_std=float(cu.std()))

        ref_u = agg["lb3"]["under_fir_mean"]
        ref_c = agg["lb3"]["clean_mean"]
        winners = [n for n in agg
                   if n != "lb3" and agg[n]["under_fir_mean"] > ref_u
                   and agg[n]["clean_mean"] >= ref_c - MARGIN]
        verdict = dict(
            pre_registered="headline lifter stays lower-bound l=3; E3 tests if any "
                           "bandpass/taper beats it under-FIR without >0.02 clean loss",
            incumbent_lb3=dict(clean=ref_c, under_fir=ref_u), margin=MARGIN,
            candidate_improvements=winners,
            conclusion=("candidate bandpass/taper improvement(s) found: "
                        + ", ".join(winners)) if winners else
                       ("no bandpass/taper beats lower-bound l=3 within margin; "
                        "the lower-bound lifter sits on the robustness plateau"),
        )
        out = dict(
            config=dict(dataset="CWRU 12k-DE 10-class", seed=SEED, n_fft=N_FFT, hop=HOP,
                        fir_sweep=FIR_SWEEP, n_splits=N_SPLITS, margin=MARGIN,
                        lifters=[dict(name=n, family=f, lo=lo, hi=hi) for n, f, lo, hi in LIFTERS]),
            lifters=agg, verdict=verdict, per_split=per_split,
        )
        (out_dir / "fir_cwru_lifter_bandpass.json").write_text(json.dumps(out, indent=2))

        print("\n=== CWRU bandpass/taper lifter sweep ===")
        for name, *_ in LIFTERS:
            a = agg[name]
            print(f"  {name:13s} clean={a['clean_mean']:.4f}  under_FIR={a['under_fir_mean']:.4f}")
        print(f"  {verdict['conclusion']}")
        print(f"  Saved {out_dir / 'fir_cwru_lifter_bandpass.json'}")


if __name__ == "__main__":
    main()
