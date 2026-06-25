"""Revision E4 — drop-one descriptor ablation of the liftered-cepstral DB-MSED.

Reviewer (DSP, 2026) Q6: "Are the 11 descriptors all necessary? An ablation
could reveal whether certain moments dominate robustness and whether
dimensionality can be reduced."

We extract the headline cepstral (l=3) 11-descriptor features ONCE per split
(train clean + test per FIR severity), then re-fit the DB-MSED head on each
leave-one-descriptor-out 10-subset and on a forward-selected top-k subset. Cost
is dominated by feature extraction (done once); the ablation is cheap re-fits.

This is DESCRIPTIVE (no pass/fail gate): it reports, per descriptor, the change
in mean under-FIR macro-F1 when it is removed, plus a reduced-set check. Splits,
seeds, and FIR channels are identical to the headline run.

Discipline: leakage-safe (file, block) groups; FIR channels/splits identical to
the headline; 15 repeated splits; under-FIR = mean over L>=3.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_TOOLKIT = Path(__file__).resolve().parents[1] / "toolkit"
if str(_TOOLKIT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT))

from run_fir_cwru_cv import (                                    # noqa: E402
    build_cwru_windows, make_real_fir, apply_real_fir, _c2_lr_f1, _cep_descriptors,
    _under, FS, N_FFT, HOP, FIR_SWEEP, PREREG_LIFT, N_SPLITS, TEST_SIZE, SEED,
)
from features import stft_cepstrum                              # noqa: E402
from run_drilling import STAT_NAMES                             # noqa: E402
from splits import grouped_split                                # noqa: E402


def cep_l3_S(x: np.ndarray) -> np.ndarray:
    cep = stft_cepstrum(x, fs=FS, n_fft=N_FFT, hop=HOP, lifter_lo=PREREG_LIFT)
    return _cep_descriptors(cep)


def _frames(windows, idx, y, fir_taps):
    Ss, ys = [], []
    for i in idx:
        xx = windows[i] if fir_taps is None else apply_real_fir(windows[i], fir_taps[i])
        S = cep_l3_S(xx)
        Ss.append(S)
        ys.append(np.full(S.shape[0], y[i], np.int64))
    return np.concatenate(Ss), np.concatenate(ys)


def _under_for_cols(S_tr, y_tr, te_feats, cols):
    """under-FIR macro-F1 using only descriptor columns `cols`."""
    sweep = {}
    for L in FIR_SWEEP:
        S_te, y_te = te_feats[L]
        sweep[L] = _c2_lr_f1(S_tr[:, cols], y_tr, S_te[:, cols], y_te)
    return _under(sweep)


def eval_one_split(windows, y, g, rs: int) -> dict:
    tr, te = grouped_split(len(windows), groups=g, test_size=TEST_SIZE, random_state=rs)
    rng = np.random.default_rng(rs + 7)
    chan = {L: {int(i): make_real_fir(L, rng) for i in te} for L in FIR_SWEEP if L > 1}

    S_tr, y_tr = _frames(windows, tr, y, None)
    te_feats = {L: _frames(windows, te, y, None if L == 1 else chan[L]) for L in FIR_SWEEP}

    d = len(STAT_NAMES)
    full = _under_for_cols(S_tr, y_tr, te_feats, list(range(d)))
    drop_one = {STAT_NAMES[j]: _under_for_cols(S_tr, y_tr, te_feats,
                                               [k for k in range(d) if k != j])
                for j in range(d)}
    # greedy forward selection on this split's under-FIR
    remaining = list(range(d))
    chosen, fwd = [], {}
    while remaining:
        best_j = max(remaining, key=lambda j: _under_for_cols(S_tr, y_tr, te_feats, chosen + [j]))
        chosen.append(best_j)
        remaining.remove(best_j)
        fwd[len(chosen)] = (_under_for_cols(S_tr, y_tr, te_feats, list(chosen)),
                            [STAT_NAMES[k] for k in chosen])
    return dict(full=full, drop_one=drop_one, forward={k: v[0] for k, v in fwd.items()},
                forward_order=[STAT_NAMES[k] for k in chosen])


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
            print(f"  split {i:2d}: full_underFIR={r['full']:.4f}", flush=True)

        full = np.array([r["full"] for r in per_split])
        drop = {nm: np.array([r["drop_one"][nm] for r in per_split]) for nm in STAT_NAMES}
        # delta = (under-FIR with descriptor removed) - full; negative => descriptor helps
        delta = {nm: float((drop[nm] - full).mean()) for nm in STAT_NAMES}
        ranked = sorted(STAT_NAMES, key=lambda nm: delta[nm])  # most-harmful-to-remove first

        kmax = len(STAT_NAMES)
        fwd_curve = {k: float(np.mean([r["forward"][k] for r in per_split]))
                     for k in range(1, kmax + 1)}

        out = dict(
            config=dict(dataset="CWRU 12k-DE 10-class", seed=SEED, n_fft=N_FFT, hop=HOP,
                        lifter=PREREG_LIFT, fir_sweep=FIR_SWEEP, n_splits=N_SPLITS,
                        descriptors=STAT_NAMES, comparator="cepstral l=3 (identical descriptors)"),
            full_under_fir_mean=float(full.mean()), full_under_fir_std=float(full.std()),
            drop_one_under_fir={nm: dict(mean=float(drop[nm].mean()),
                                         delta_vs_full=delta[nm]) for nm in STAT_NAMES},
            ranked_by_importance=ranked,
            forward_selection_under_fir=fwd_curve,
            forward_order_modal=per_split[0]["forward_order"],
        )
        (out_dir / "fir_cwru_ablation.json").write_text(json.dumps(out, indent=2))

        print("\n=== CWRU descriptor ablation (cepstral l=3, under FIR) ===")
        print(f"  full 11-descriptor under_FIR = {full.mean():.4f}")
        print("  drop-one delta (negative => descriptor contributes robustness):")
        for nm in ranked:
            print(f"    {nm:9s} drop->under_FIR={drop[nm].mean():.4f}  delta={delta[nm]:+.4f}")
        print("  forward-selection under_FIR by k:",
              {k: round(v, 4) for k, v in fwd_curve.items()})
        print(f"  Saved {out_dir / 'fir_cwru_ablation.json'}")


if __name__ == "__main__":
    main()
