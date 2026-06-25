"""Cross-modal HARDENING of the cepstral convolutional-robustness thesis on CWRU
bearing-fault vibration (B1 + B2 of ROADMAP).

Motivation
----------
The single-split DroneRF FIR gate (results/fir_gate/fir_lifter_sweep.json) found a
lifter sweet-spot (lift_lo = 3) where a per-class Kunchenko-space (C2) classifier on
the real cepstrum BEATS the same classifier on the log-spectral profile under injected
FIR multipath — but on ONE leakage-safe split, ~1320 RF-2G windows, a modest +0.04
margin. The roadmap flags two gaps before the positive claim is paper-grade:
  B1 — repeated-split + bootstrap confirmation;
  B2 — a second modality, for a cross-domain robustness claim.
This script closes both at once on CWRU 12 kHz drive-end vibration (a genuinely
convolutive domain: the bearing→sensor mechanical transfer path is an FIR filter).

PRE-REGISTERED criterion  (fixed BEFORE running on CWRU — carried VERBATIM from the RF
experiment; NOT tuned to CWRU; this is a transfer test, not a selection):
  At the RF-derived sweet-spot lifter (lift_lo = PREREG_LIFT = 3) the cepstral-C2
  classifier's mean macro-F1 UNDER FIR multipath (severities L >= 3) exceeds the
  spectral-C2 classifier's mean macro-F1 under FIR, and the per-split gap
  (cepstral_lift3_underFIR − spectral_underFIR) is:
     (i)  positive under a one-sided Wilcoxon signed-rank test, p < 0.05, AND
     (ii) has a 95% bootstrap CI (R = 2000) on its mean that excludes zero,
  across N_SPLITS = 15 leakage-safe repeated splits.
The full lifter grid [1,2,3,4,6,8] is reported for the clean-vs-robust trade-off curve;
the HEADLINE decision uses only the pre-registered lift_lo = 3. A "best lifter per split"
view is reported separately and labelled as optimistic (post-hoc) for transparency.

Discipline honoured: leakage-safe (file, block) groups with a raw-sample guard; equal
feature budget (both representations feed 11 shape descriptors → C2); nested/fixed
selection (lifter fixed a-priori from RF); repeated-split CV + bootstrap before the claim.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_TOOLKIT = Path(__file__).resolve().parents[1] / "toolkit"
if str(_TOOLKIT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT))

from features import stft_log_profile, stft_cepstrum          # noqa: E402
from h2_equal_budget import spectral_statistics_extended       # noqa: E402
from run_drilling import shape_descriptors_1ch, STAT_NAMES     # noqa: E402
from stats_basis import c2_kunchenko_mahalanobis               # noqa: E402
from splits import grouped_split                               # noqa: E402

# ----------------------------- configuration ------------------------------- #
SEED = 2026
FS = 12000.0
WIN = 4096
N_FFT = 256
HOP = 128
K_PER_BLOCK = 4
GUARD = WIN                 # raw samples dropped between blocks → no overlap
MAX_WIN_PER_CLASS = 120
LIFTERS = [1, 2, 3, 4, 6, 8]
PREREG_LIFT = 3             # transferred from the RF sweet-spot — NOT tuned on CWRU
FIR_SWEEP = [1, 2, 3, 5, 8, 12]   # L = 1 ⇒ clean (identity); L = taps for multipath
N_SPLITS = 15
TEST_SIZE = 0.3
N_BOOT = 2000

# Descriptor pipeline for the SPECTRAL branch (revision E0). The reviewer-facing
# claim is a *representation* effect, so both branches must share one descriptor
# function. "identical" feeds the spectral log-profile through the SAME per-frame
# z-norm + shape_descriptors_1ch path as the cepstral branch (true representation
# isolation; the revision headline). "branch" reproduces the as-submitted numbers,
# where the spectral branch used spectral_statistics_extended instead.
DESC_MODE = os.environ.get("DESC_MODE", "identical")

_CWRU = _HERE.parent / "data" / "cwru"

# CWRU 12k drive-end, file → (label, class_id) — same 10-class map as run_cwru.py.
FILE_MAP = {
    **{n: ("Normal", 0) for n in (97, 98, 99, 100)},
    **{n: ("IR007", 1) for n in (105, 106, 107, 108)},
    **{n: ("IR014", 2) for n in (169, 170, 171, 172)},
    **{n: ("IR021", 3) for n in (209, 210, 211, 212)},
    **{n: ("Ball007", 4) for n in (118, 119, 120, 121)},
    **{n: ("Ball014", 5) for n in (185, 186, 187, 188)},
    **{n: ("Ball021", 6) for n in (222, 223, 224, 225)},
    **{n: ("OR007", 7) for n in (130, 131, 132, 133)},
    **{n: ("OR014", 8) for n in (197, 198, 199, 200)},
    **{n: ("OR021", 9) for n in (234, 235, 236, 237)},
}
CLASS_NAMES = ["Normal", "IR007", "IR014", "IR021", "Ball007", "Ball014",
               "Ball021", "OR007", "OR014", "OR021"]


# ----------------------------- data loading -------------------------------- #
def _load_de(n: int) -> np.ndarray:
    m = sio.loadmat(str(_CWRU / f"{n}.mat"))
    de = [k for k in m if k.endswith("_DE_time")]
    if not de:
        raise RuntimeError(f"{n}.mat has no *_DE_time variable")
    return m[de[0]].ravel().astype(np.float64)


def build_cwru_windows():
    """Raw non-overlapping WIN-sample windows; leakage-safe (file, block) groups.

    Windows within a block are contiguous and non-overlapping; a GUARD of WIN
    samples is dropped between blocks so no two blocks share raw samples. The
    group id is unique per (file, block), and a grouped split keeps whole blocks
    together — frames produced later from any window never cross train/test.
    """
    windows, ys, gs = [], [], []
    per_class: dict[int, int] = defaultdict(int)
    gid = 0
    for n, (label, cid) in sorted(FILE_MAP.items()):
        sig = _load_de(n)
        pos = 0
        while pos + K_PER_BLOCK * WIN <= len(sig) and per_class[cid] < MAX_WIN_PER_CLASS:
            for k in range(K_PER_BLOCK):
                if per_class[cid] >= MAX_WIN_PER_CLASS:
                    break
                s = pos + k * WIN
                windows.append(sig[s:s + WIN].copy())
                ys.append(cid)
                gs.append(gid)
                per_class[cid] += 1
            pos += K_PER_BLOCK * WIN + GUARD
            gid += 1
    return windows, np.array(ys), np.array(gs), CLASS_NAMES


# --------------------------- FIR multipath channel ------------------------- #
def make_real_fir(n_taps: int, rng: np.random.Generator, decay: float = 0.7) -> np.ndarray:
    """Real-valued multipath FIR (mechanical transmission path), unit L2 energy.

    Real analogue of synthetic.make_fir_channel: tap 0 = dominant direct path,
    later taps = decaying random echoes. n_taps = 1 ⇒ identity (clean point).
    """
    if n_taps <= 1:
        return np.array([1.0], dtype=np.float64)
    k = np.arange(n_taps)
    amp = np.exp(-decay * k)
    taps = amp * rng.standard_normal(n_taps)
    taps[0] = amp[0]                       # dominant, sign-aligned direct path
    taps = taps / np.sqrt(np.sum(taps ** 2))
    return taps.astype(np.float64)


def apply_real_fir(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    if taps.size <= 1:
        return (x * taps[0]).astype(np.float64, copy=False)
    return np.convolve(x, taps, mode="full")[:len(x)].astype(np.float64, copy=False)


# --------------------------- feature extraction ---------------------------- #
def spectral_S(x: np.ndarray) -> np.ndarray:
    prof = stft_log_profile(x, fs=FS, n_fft=N_FFT, hop=HOP)
    if DESC_MODE == "identical":
        # Same per-frame z-norm + shape_descriptors_1ch as the cepstral branch,
        # so the spectral-vs-cepstral comparison isolates the representation.
        return _cep_descriptors(prof)
    return spectral_statistics_extended(prof)


def _cep_descriptors(cep: np.ndarray) -> np.ndarray:
    """Per-frame 11-d shape descriptors of a (liftered) real cepstrum block."""
    out = np.empty((cep.shape[0], len(STAT_NAMES)), dtype=np.float64)
    for i in range(cep.shape[0]):
        v = cep[i]
        s = v.std()
        vz = (v - v.mean()) / s if s > 1e-9 else v - v.mean()
        out[i] = shape_descriptors_1ch(vz)
    return out


def spec_frames(windows, idx, y, fir_taps):
    Ss, ys = [], []
    for i in idx:
        x = windows[i] if fir_taps is None else apply_real_fir(windows[i], fir_taps[i])
        S = spectral_S(x)
        Ss.append(S)
        ys.append(np.full(S.shape[0], y[i], np.int64))
    return np.concatenate(Ss), np.concatenate(ys)


def cep_frames_all_lifters(windows, idx, y, fir_taps):
    """One STFT-cepstrum pass per window; derive ALL lifters by column slicing.

    stft_cepstrum(lifter_lo=1) returns cepstrum columns at quefrency 1..hi-1;
    lifter_lo = L is exactly that array sliced from column (L-1).
    """
    per_l = {L: ([], []) for L in LIFTERS}
    for i in idx:
        x = windows[i] if fir_taps is None else apply_real_fir(windows[i], fir_taps[i])
        cep_full = stft_cepstrum(x, fs=FS, n_fft=N_FFT, hop=HOP, lifter_lo=1)
        for L in LIFTERS:
            cep = cep_full[:, (L - 1):]
            S = _cep_descriptors(cep)
            per_l[L][0].append(S)
            per_l[L][1].append(np.full(S.shape[0], y[i], np.int64))
    return {L: (np.concatenate(a), np.concatenate(b)) for L, (a, b) in per_l.items()}


def _c2_lr_f1(S_tr, y_tr, S_te, y_te) -> float:
    c2_tr, c2_te = c2_kunchenko_mahalanobis(S_tr, y_tr, S_te)
    sc = StandardScaler().fit(c2_tr)
    clf = LogisticRegression(max_iter=2000, n_jobs=1).fit(sc.transform(c2_tr), y_tr)
    return float(f1_score(y_te, clf.predict(sc.transform(c2_te)), average="macro"))


def _under(sweep: dict) -> float:
    return float(np.mean([sweep[L] for L in FIR_SWEEP if L >= 3]))


# ------------------------------ one split ---------------------------------- #
def eval_one_split(windows, y, g, rs: int) -> dict:
    tr, te = grouped_split(len(windows), groups=g, test_size=TEST_SIZE, random_state=rs)
    rng = np.random.default_rng(rs + 7)
    chan = {L: {int(i): make_real_fir(L, rng) for i in te} for L in FIR_SWEEP if L > 1}

    # spectral: train on clean, test under each FIR severity
    Sspec_tr, yspec_tr = spec_frames(windows, tr, y, None)
    spec_sweep = {}
    for L in FIR_SWEEP:
        taps = None if L == 1 else chan[L]
        Ste, yte = spec_frames(windows, te, y, taps)
        spec_sweep[L] = _c2_lr_f1(Sspec_tr, yspec_tr, Ste, yte)

    # cepstral: train features per lifter (one pass), test per (FIR severity)
    cep_tr = cep_frames_all_lifters(windows, tr, y, None)
    cep_te = {L: {} for L in FIR_SWEEP}
    for L in FIR_SWEEP:
        taps = None if L == 1 else chan[L]
        cep_te[L] = cep_frames_all_lifters(windows, te, y, taps)

    cep_sweep = {}
    for Llift in LIFTERS:
        Str, ytr = cep_tr[Llift]
        sw = {}
        for Lfir in FIR_SWEEP:
            Ste, yte = cep_te[Lfir][Llift]
            sw[Lfir] = _c2_lr_f1(Str, ytr, Ste, yte)
        cep_sweep[Llift] = sw

    spec_under = _under(spec_sweep)
    cep_under = {L: _under(cep_sweep[L]) for L in LIFTERS}
    best_lift = max(LIFTERS, key=lambda L: cep_under[L])
    return dict(
        rs=rs,
        spectral=dict(clean=spec_sweep[1], under_fir=spec_under,
                      sweep={int(L): spec_sweep[L] for L in FIR_SWEEP}),
        cepstral={int(L): dict(clean=cep_sweep[L][1], under_fir=cep_under[L],
                               sweep={int(Lf): cep_sweep[L][Lf] for Lf in FIR_SWEEP})
                  for L in LIFTERS},
        gap_prereg=float(cep_under[PREREG_LIFT] - spec_under),
        best_lift=int(best_lift),
        gap_best=float(cep_under[best_lift] - spec_under),
    )


def _boot_ci(vals: np.ndarray, rng: np.random.Generator, R: int = N_BOOT):
    means = np.empty(R)
    n = len(vals)
    for r in range(R):
        means[r] = vals[rng.integers(0, n, n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    out_dir = _HERE.parent / "results" / "fir_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(">>> building CWRU raw windows", flush=True)
        windows, y, g, names = build_cwru_windows()
        print(f"windows={len(windows)}, classes={len(names)}, "
              f"groups={len(np.unique(g))}", flush=True)

        per_split = []
        for i in range(N_SPLITS):
            r = eval_one_split(windows, y, g, rs=SEED + i)
            per_split.append(r)
            print(f"  split {i:2d}: spec_underFIR={r['spectral']['under_fir']:.4f} "
                  f"cep(L{PREREG_LIFT})_underFIR={r['cepstral'][PREREG_LIFT]['under_fir']:.4f} "
                  f"gap={r['gap_prereg']:+.4f} (best L{r['best_lift']} gap={r['gap_best']:+.4f})",
                  flush=True)

        gaps = np.array([r["gap_prereg"] for r in per_split])
        spec_under = np.array([r["spectral"]["under_fir"] for r in per_split])
        cep_under = np.array([r["cepstral"][PREREG_LIFT]["under_fir"] for r in per_split])
        spec_clean = np.array([r["spectral"]["clean"] for r in per_split])
        cep_clean = np.array([r["cepstral"][PREREG_LIFT]["clean"] for r in per_split])

        w = sps.wilcoxon(cep_under, spec_under, alternative="greater")
        rng = np.random.default_rng(SEED + 99)
        ci_lo, ci_hi = _boot_ci(gaps, rng)

        passed = bool((w.pvalue < 0.05) and (ci_lo > 0))
        verdict = dict(
            pre_registered_criterion=(
                "At lift_lo=3 (transferred from RF, not tuned on CWRU): cepstral-C2 mean "
                "macro-F1 under FIR (L>=3) > spectral-C2 under FIR, with one-sided Wilcoxon "
                "p<0.05 AND 95% bootstrap CI on the per-split gap excluding zero, over 15 "
                "leakage-safe repeated splits."),
            prereg_lift=PREREG_LIFT,
            n_splits=N_SPLITS,
            gap_mean=float(gaps.mean()), gap_std=float(gaps.std()),
            gap_min=float(gaps.min()), gap_max=float(gaps.max()),
            frac_wins=float((gaps > 0).mean()),
            wilcoxon_stat=float(w.statistic), wilcoxon_p=float(w.pvalue),
            bootstrap_ci95=[ci_lo, ci_hi],
            passed=passed,
            verdict=("PASS — cepstral convolutional-robustness sweet-spot transfers to a "
                     "second modality (vibration) with repeated-split + bootstrap support")
                    if passed else
                    ("FAIL — the RF sweet-spot does not transfer to CWRU under the "
                     "pre-registered test; report as honest cross-domain limit"),
        )

        # lifter-sweep means across splits — for the trade-off curve figure
        lifter_curve = {}
        for L in LIFTERS:
            cu = np.array([r["cepstral"][L]["under_fir"] for r in per_split])
            cc = np.array([r["cepstral"][L]["clean"] for r in per_split])
            lifter_curve[int(L)] = dict(
                clean_mean=float(cc.mean()), clean_std=float(cc.std()),
                under_fir_mean=float(cu.mean()), under_fir_std=float(cu.std()),
                beats_spectral_under_fir=bool(cu.mean() > spec_under.mean()))

        out = dict(
            config=dict(dataset="CWRU 12k-DE 10-class (type×diameter)", seed=SEED,
                        fs=FS, win=WIN, n_fft=N_FFT, hop=HOP, k_per_block=K_PER_BLOCK,
                        guard=GUARD, max_win_per_class=MAX_WIN_PER_CLASS,
                        lifters=LIFTERS, prereg_lift=PREREG_LIFT, fir_sweep=FIR_SWEEP,
                        n_splits=N_SPLITS, test_size=TEST_SIZE, n_boot=N_BOOT,
                        n_windows=len(windows), n_groups=int(len(np.unique(g))),
                        classes=names, desc_mode=DESC_MODE,
                        equal_feature_budget=(
                            "11 identical z-norm shape descriptors (both branches) → C2"
                            if DESC_MODE == "identical"
                            else "11 descriptors → C2 (branch-specific: spectral_statistics_extended)")),
            aggregate=dict(
                spectral=dict(clean_mean=float(spec_clean.mean()), clean_std=float(spec_clean.std()),
                              under_fir_mean=float(spec_under.mean()), under_fir_std=float(spec_under.std())),
                cepstral_prereg=dict(lift=PREREG_LIFT,
                                     clean_mean=float(cep_clean.mean()), clean_std=float(cep_clean.std()),
                                     under_fir_mean=float(cep_under.mean()), under_fir_std=float(cep_under.std())),
                lifter_curve=lifter_curve),
            verdict=verdict,
            per_split=per_split,
        )
        out_name = "fir_cwru_cv.json" if DESC_MODE == "identical" else "fir_cwru_cv_branch.json"
        out_path = out_dir / out_name
        out_path.write_text(json.dumps(out, indent=2))

        print(f"\n=== CWRU FIR-robustness repeated-split CV (desc_mode={DESC_MODE}) ===")
        print(f"  spectral        clean={spec_clean.mean():.4f}±{spec_clean.std():.4f}  "
              f"under_FIR={spec_under.mean():.4f}±{spec_under.std():.4f}")
        print(f"  cepstral lift={PREREG_LIFT}  clean={cep_clean.mean():.4f}±{cep_clean.std():.4f}  "
              f"under_FIR={cep_under.mean():.4f}±{cep_under.std():.4f}")
        print(f"  gap (cep_underFIR − spec_underFIR): mean={gaps.mean():+.4f}  "
              f"wins={int((gaps>0).sum())}/{N_SPLITS}  Wilcoxon p={w.pvalue:.2e}  "
              f"boot95=[{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"  VERDICT: {verdict['verdict']}")
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
