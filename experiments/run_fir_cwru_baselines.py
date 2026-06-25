"""Revision E1+E2 — matched-budget baselines for the cepstral FIR-robustness claim.

Reviewer (DSP, 2026) asked for two things the original submission lacked:
  E1. A SPECTRAL channel-mitigation control: does a cheap, classical channel
      remover on the spectral profile (CMVN / log-spectrum polynomial detrend /
      RASTA) already match the liftered cepstrum under FIR? If so, the gain is
      "just channel-mean removal" and must be reported as such.
  E2. A matched-budget MFCC baseline: MFCC(11) through the SAME DB-MSED head
      (apples-to-apples representation swap), and MFCC through the literature
      heads (SVM, per-class GMM).

All methods use the SAME 11-dim feature budget and, where noted, the SAME
DB-MSED head (per-class regularised Mahalanobis -> StandardScaler -> LR) as the
headline. Channels, splits, and seeds are byte-identical to run_fir_cwru_cv.py
(DESC_MODE=identical headline), so the per-split comparison against the stored
cepstral l=3 numbers is a valid paired test.

PRE-REGISTERED criterion (fixed BEFORE running):
  At matched 11-budget and the identical DB-MSED head, the liftered cepstrum
  (l=3) mean macro-F1 UNDER FIR (L>=3) exceeds the BEST spectral
  channel-mitigation control (CMVN, polynomial-detrend, RASTA), with the
  per-split gap (cep_l3 - best_control):
     (i)  positive under a one-sided Wilcoxon signed-rank test, p < 0.05, AND
     (ii) 95% bootstrap CI (R = 2000) on its mean excluding zero,
  across N_SPLITS = 15 leakage-safe splits.
If a spectral control matches or beats the cepstrum, that is reported as the
honest finding (the reviewer's exact concern), NOT softened.
The MFCC heads are reported (clean + under-FIR) without a pass/fail gate; all
heads are reported, none is cherry-picked.

Discipline: leakage-safe (file, block) groups; equal 11-feature budget; FIR
channels and splits identical to the headline; repeated-split CV + bootstrap.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.fftpack import dct
from scipy.signal import lfilter, stft as scipy_stft
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_TOOLKIT = Path(__file__).resolve().parents[1] / "toolkit"
if str(_TOOLKIT) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT))

# Reuse the headline harness verbatim (DESC_MODE defaults to "identical").
from run_fir_cwru_cv import (                                    # noqa: E402
    build_cwru_windows, make_real_fir, apply_real_fir, _boot_ci, _c2_lr_f1,
    _cep_descriptors, _under, FS, WIN, N_FFT, HOP, FIR_SWEEP, PREREG_LIFT,
    N_SPLITS, TEST_SIZE, SEED,
)
from features import stft_log_profile                            # noqa: E402
from splits import grouped_split                                 # noqa: E402

_EPS = 1e-8


# ---------------------- E1: spectral channel-mitigation -------------------- #
def _profile(x: np.ndarray) -> np.ndarray:
    return stft_log_profile(x, fs=FS, n_fft=N_FFT, hop=HOP)


def cmvn_spectral_S(x: np.ndarray) -> np.ndarray:
    """CMVN on the log-spectral profile: per-window, per-bin mean/var
    normalisation across the window's frames (removes stationary channel
    coloration without estimating the channel). Then the IDENTICAL 11 descriptors."""
    P = _profile(x)
    mu = P.mean(0, keepdims=True)
    sd = P.std(0, keepdims=True)
    return _cep_descriptors((P - mu) / (sd + _EPS))


def detrend_spectral_S(x: np.ndarray, deg: int = 3) -> np.ndarray:
    """Per-frame low-order polynomial detrend across frequency: subtract the
    smooth spectral envelope (the channel tilt). Then the IDENTICAL 11 descriptors."""
    P = _profile(x)
    n_bins = P.shape[1]
    t = np.linspace(-1.0, 1.0, n_bins)
    V = np.vander(t, deg + 1)                       # (n_bins, deg+1)
    coef, *_ = np.linalg.lstsq(V, P.T, rcond=None)  # (deg+1, n_frames)
    trend = (V @ coef).T                            # (n_frames, n_bins)
    return _cep_descriptors(P - trend)


# RASTA bandpass (Hermansky 1994) applied along the frame axis, per bin.
_RASTA_B = 0.1 * np.array([2.0, 1.0, 0.0, -1.0, -2.0])
_RASTA_A = np.array([1.0, -0.94])


def rasta_spectral_S(x: np.ndarray) -> np.ndarray:
    """RASTA-style temporal bandpass of each log-spectral bin trajectory across
    frames (de-emphasises slowly varying channel terms). Then 11 descriptors."""
    P = _profile(x)
    if P.shape[0] < 5:                              # too few frames for the IIR
        return _cep_descriptors(P)
    Pr = lfilter(_RASTA_B, _RASTA_A, P, axis=0)
    return _cep_descriptors(Pr)


# ------------------------------- E2: MFCC ---------------------------------- #
def _mel_filterbank(n_mels: int, n_fft: int, fs: float) -> np.ndarray:
    """Triangular mel filterbank over the one-sided power spectrum."""
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_one = n_fft // 2 + 1
    mels = np.linspace(hz2mel(0.0), hz2mel(fs / 2.0), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / fs).astype(int)
    bins = np.clip(bins, 0, n_one - 1)
    fb = np.zeros((n_mels, n_one))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        c = max(c, l + 1)
        r = max(r, c + 1)
        if c < n_one:
            fb[i - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        if r <= n_one:
            fb[i - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


_N_MELS = 26
_MEL = _mel_filterbank(_N_MELS, N_FFT, FS)


def mfcc_S(x: np.ndarray, n_keep: int = 11) -> np.ndarray:
    """Standard MFCC: |X|^2 -> mel filterbank -> log -> DCT-II; drop c0, keep
    n_keep coefficients (matched 11-dim budget)."""
    _, _, Z = scipy_stft(x, fs=FS, window="hann", nperseg=N_FFT,
                         noverlap=N_FFT - HOP, return_onesided=True,
                         boundary=None, padded=False)
    pow_spec = np.abs(Z) ** 2                        # (n_one, n_frames)
    mel_e = _MEL @ pow_spec                          # (n_mels, n_frames)
    log_mel = np.log(mel_e + _EPS)
    cc = dct(log_mel, type=2, axis=0, norm="ortho")  # (n_mels, n_frames)
    return cc[1:1 + n_keep, :].T                     # drop c0 -> (n_frames, 11)


# ------------------------------- heads ------------------------------------- #
def _svm_f1(S_tr, y_tr, S_te, y_te) -> float:
    sc = StandardScaler().fit(S_tr)
    clf = SVC(C=10.0, gamma="scale").fit(sc.transform(S_tr), y_tr)
    return float(f1_score(y_te, clf.predict(sc.transform(S_te)), average="macro"))


def _gmm_f1(S_tr, y_tr, S_te, y_te) -> float:
    sc = StandardScaler().fit(S_tr)
    Ztr, Zte = sc.transform(S_tr), sc.transform(S_te)
    classes = np.unique(y_tr)
    ll = np.empty((Zte.shape[0], len(classes)))
    for ci, c in enumerate(classes):
        gm = GaussianMixture(n_components=2, covariance_type="diag",
                             reg_covar=1e-3, random_state=SEED).fit(Ztr[y_tr == c])
        ll[:, ci] = gm.score_samples(Zte)
    return float(f1_score(y_te, classes[ll.argmax(1)], average="macro"))


_HEAD = {"dbmsed": _c2_lr_f1, "svm": _svm_f1, "gmm": _gmm_f1}

METHODS = {
    "cmvn_spectral":    (cmvn_spectral_S, "dbmsed", "E1"),
    "detrend_spectral": (detrend_spectral_S, "dbmsed", "E1"),
    "rasta_spectral":   (rasta_spectral_S, "dbmsed", "E1"),
    "mfcc_dbmsed":      (mfcc_S, "dbmsed", "E2"),
    "mfcc_svm":         (mfcc_S, "svm", "E2"),
    "mfcc_gmm":         (mfcc_S, "gmm", "E2"),
}


def _frames(feat_fn, windows, idx, y, fir_taps):
    Ss, ys = [], []
    for i in idx:
        xx = windows[i] if fir_taps is None else apply_real_fir(windows[i], fir_taps[i])
        S = feat_fn(xx)
        Ss.append(S)
        ys.append(np.full(S.shape[0], y[i], np.int64))
    return np.concatenate(Ss), np.concatenate(ys)


def _method_sweep(feat_fn, head, windows, tr, te, y, chan):
    """clean + per-severity under-FIR macro-F1 for one method on one split."""
    f1_head = _HEAD[head]
    S_tr, y_tr = _frames(feat_fn, windows, tr, y, None)
    sweep = {}
    for L in FIR_SWEEP:
        taps = None if L == 1 else chan[L]
        S_te, y_te = _frames(feat_fn, windows, te, y, taps)
        sweep[L] = f1_head(S_tr, y_tr, S_te, y_te)
    return dict(clean=sweep[1], under_fir=_under(sweep),
                sweep={int(L): sweep[L] for L in FIR_SWEEP})


def eval_one_split(windows, y, g, rs: int) -> dict:
    # EXACT replica of run_fir_cwru_cv.eval_one_split channel construction so the
    # injected FIR channels match the headline run for a valid paired comparison.
    tr, te = grouped_split(len(windows), groups=g, test_size=TEST_SIZE, random_state=rs)
    rng = np.random.default_rng(rs + 7)
    chan = {L: {int(i): make_real_fir(L, rng) for i in te} for L in FIR_SWEEP if L > 1}
    out = {}
    for name, (fn, head, _tier) in METHODS.items():
        out[name] = _method_sweep(fn, head, windows, tr, te, y, chan)
    return out


def main():
    out_dir = _HERE.parent / "results" / "fir_gate"
    head_path = out_dir / "fir_cwru_cv.json"
    if not head_path.exists():
        raise SystemExit("run_fir_cwru_cv.py (DESC_MODE=identical) must run first "
                         "to produce the cepstral l=3 comparator.")
    headline = json.loads(head_path.read_text())
    cep_under = np.array([r["cepstral"][str(PREREG_LIFT)]["under_fir"]
                          if str(PREREG_LIFT) in r["cepstral"]
                          else r["cepstral"][PREREG_LIFT]["under_fir"]
                          for r in headline["per_split"]])
    cep_clean = np.array([r["cepstral"][str(PREREG_LIFT)]["clean"]
                          if str(PREREG_LIFT) in r["cepstral"]
                          else r["cepstral"][PREREG_LIFT]["clean"]
                          for r in headline["per_split"]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(">>> building CWRU raw windows", flush=True)
        windows, y, g, names = build_cwru_windows()
        print(f"windows={len(windows)}, classes={len(names)}", flush=True)

        per_split = []
        for i in range(N_SPLITS):
            r = eval_one_split(windows, y, g, rs=SEED + i)
            per_split.append(r)
            tags = "  ".join(f"{m[:10]}={r[m]['under_fir']:.3f}" for m in METHODS)
            print(f"  split {i:2d}: cep_l3={cep_under[i]:.3f} | {tags}", flush=True)

        # aggregate per method
        agg = {}
        for name in METHODS:
            cu = np.array([r[name]["under_fir"] for r in per_split])
            cc = np.array([r[name]["clean"] for r in per_split])
            agg[name] = dict(tier=METHODS[name][2], head=METHODS[name][1],
                             clean_mean=float(cc.mean()), clean_std=float(cc.std()),
                             under_fir_mean=float(cu.mean()), under_fir_std=float(cu.std()))

        # E1 head-to-head: cepstral l=3 vs BEST spectral channel-mitigation control
        from scipy import stats as sps
        e1 = ["cmvn_spectral", "detrend_spectral", "rasta_spectral"]
        ctrl_under = {m: np.array([r[m]["under_fir"] for r in per_split]) for m in e1}
        best_ctrl = max(e1, key=lambda m: ctrl_under[m].mean())
        gap = cep_under - ctrl_under[best_ctrl]
        w = sps.wilcoxon(cep_under, ctrl_under[best_ctrl], alternative="greater")
        rng = np.random.default_rng(SEED + 99)
        ci_lo, ci_hi = _boot_ci(gap, rng)
        passed = bool((w.pvalue < 0.05) and (ci_lo > 0))
        verdict = dict(
            pre_registered_criterion=(
                "cepstral l=3 mean under-FIR macro-F1 > BEST spectral channel-mitigation "
                "control (CMVN / poly-detrend / RASTA) at matched 11-budget + identical "
                "DB-MSED head, one-sided Wilcoxon p<0.05 AND 95% bootstrap CI on the "
                "per-split gap excluding zero, over 15 leakage-safe splits."),
            best_control=best_ctrl,
            cep_l3_under_fir_mean=float(cep_under.mean()),
            best_control_under_fir_mean=float(ctrl_under[best_ctrl].mean()),
            gap_mean=float(gap.mean()), gap_min=float(gap.min()), gap_max=float(gap.max()),
            frac_wins=float((gap > 0).mean()),
            wilcoxon_p=float(w.pvalue), bootstrap_ci95=[ci_lo, ci_hi], passed=passed,
            verdict=("PASS — liftered cepstrum beats the best cheap spectral channel "
                     "remover under FIR; the gain is not reducible to channel-mean removal")
                    if passed else
                    ("INCONCLUSIVE/FAIL — a cheap spectral control is competitive with the "
                     "cepstrum under FIR; report honestly and re-scope the contribution"),
        )

        out = dict(
            config=dict(dataset="CWRU 12k-DE 10-class", seed=SEED, fs=FS, win=WIN,
                        n_fft=N_FFT, hop=HOP, fir_sweep=FIR_SWEEP, n_splits=N_SPLITS,
                        test_size=TEST_SIZE, n_mels=_N_MELS, mfcc_keep=11, detrend_deg=3,
                        comparator="cepstral l=3 (DESC_MODE=identical headline)",
                        equal_feature_budget="11 features for every method"),
            cepstral_l3=dict(clean_mean=float(cep_clean.mean()),
                             under_fir_mean=float(cep_under.mean())),
            methods=agg, e1_verdict=verdict, per_split=per_split,
        )
        out_path = out_dir / "fir_cwru_baselines.json"
        out_path.write_text(json.dumps(out, indent=2))

        print(f"\n=== CWRU FIR baselines (matched 11-budget) ===")
        print(f"  cepstral l=3     clean={cep_clean.mean():.4f}  under_FIR={cep_under.mean():.4f}")
        for name in METHODS:
            a = agg[name]
            print(f"  {name:17s}[{a['tier']}/{a['head']:6s}] clean={a['clean_mean']:.4f}  "
                  f"under_FIR={a['under_fir_mean']:.4f}")
        print(f"  E1 best control = {best_ctrl}: gap(cep_l3 - ctrl) under FIR "
              f"mean={gap.mean():+.4f}  wins={int((gap>0).sum())}/{N_SPLITS}  "
              f"Wilcoxon p={w.pvalue:.2e}  boot95=[{ci_lo:+.4f},{ci_hi:+.4f}]")
        print(f"  VERDICT: {verdict['verdict']}")
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
