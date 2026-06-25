"""Spectral profile and feature extraction for DSGE-Spectral H1-H4.

Three feature views, all derived from the per-frame log-magnitude STFT profile:
 1. ``stft_log_profile`` — energy-normalised log-|STFT|² per frame, used as the
    raw input to DSGE (Scenario A: across-bin distribution within a frame).
 2. ``spectral_statistics`` — fixed across-bin descriptors (flatness, entropy,
    centroid, skewness, kurtosis). Baseline that H2 must beat.
 3. ``flatten_for_position`` — raw per-bin log-magnitudes; ``PCA(20)`` later
    gives the position-aware feature view (H3/H4 comparator). Naturally moves
    under CFO since each bin is location-coded.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft as scipy_stft


_LOG_EPS = 1e-12


def stft_log_profile(
    x: np.ndarray,
    fs: float,
    n_fft: int = 256,
    hop: int = 128,
    window: str = "hann",
    normalise_per_frame: bool = True,
) -> np.ndarray:
    """Compute log-magnitude STFT profile per frame.

    Parameters
    ----------
    x  : complex baseband signal, shape (n_samples,).
    fs : sample rate (Hz).
    n_fft, hop, window : STFT params.
    normalise_per_frame : if True, subtract per-frame mean of the log spectrum
        — equivalent to dividing |X|² by per-frame geometric mean magnitude²
        (per spec §6: per-record energy normalisation so DSGE learns shape not
        loudness; we do it per-frame which is stricter and matches the across-
        frame STFT convention).

    Returns
    -------
    profile : (n_frames, n_bins) float64, where each row is the log-magnitude
              profile of one frame, with the DC bin dropped (it carries no
              shape information after normalisation).
    """
    f, t, Z = scipy_stft(
        x,
        fs=fs,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    # Z: (n_bins, n_frames). Drop DC bin (index 0) — it absorbs the energy mean.
    Z = Z[1:, :]
    mag2 = np.abs(Z) ** 2
    log_p = np.log(mag2 + _LOG_EPS)
    if normalise_per_frame:
        log_p = log_p - log_p.mean(axis=0, keepdims=True)
    return log_p.T.astype(np.float64)  # (n_frames, n_bins)


def stft_cepstrum(
    x: np.ndarray,
    fs: float,
    n_fft: int = 256,
    hop: int = 128,
    window: str = "hann",
    lifter_lo: int = 0,
    lifter_hi: int | None = None,
) -> np.ndarray:
    """Real cepstrum per STFT frame: c[q] = Re IDFT( log(|X[k]| + eps) ).

    The homomorphic-deconvolution property: a convolutional channel `h`
    (`y = x * h`) adds `log|H|` to the log-spectrum, which is **smooth** and so
    concentrates at **low quefrency** in the cepstrum (`c_y = c_x + c_h`,
    `c_h` low-q). Liftering out the lowest quefrencies (``lifter_lo``) therefore
    removes the channel — the basis of the convolutional-robustness thesis.

    Parameters
    ----------
    lifter_lo : drop quefrency indices `< lifter_lo` (index 0 is log-energy;
        low indices hold the smooth envelope/channel).
    lifter_hi : keep quefrency indices `< lifter_hi` (default n_fft//2 — the
        non-redundant half of the real cepstrum).

    Returns
    -------
    cep : (n_frames, n_quef_kept) real cepstrum, liftered to
          [lifter_lo : lifter_hi).
    """
    f, t, Z = scipy_stft(
        x, fs=fs, window=window, nperseg=n_fft, noverlap=n_fft - hop,
        return_onesided=False, boundary=None, padded=False,
    )
    # Z: (n_bins, n_frames). Full two-sided log-magnitude → real cepstrum.
    log_mag = np.log(np.abs(Z) + 1e-8)               # (n_fft, n_frames)
    cep = np.fft.ifft(log_mag, axis=0).real          # (n_fft, n_frames)
    hi = (n_fft // 2) if lifter_hi is None else lifter_hi
    lo = max(1, lifter_lo)  # always drop c[0] (log-energy / loudness)
    return cep[lo:hi, :].T.astype(np.float64)        # (n_frames, hi-lo)


def stack_signal_profiles(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    fs: float, n_fft: int = 256, hop: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply STFT to every signal and produce a frame-level dataset.

    Returns ``(profiles, y_frame, group_frame)`` where:
     - profiles    : (n_total_frames, n_bins) — one row per frame, across all
       signals;
     - y_frame     : (n_total_frames,) — class label of each frame's parent
       signal;
     - group_frame : (n_total_frames,) — signal-id of each frame (for the
       leakage-safe split that keeps frames from one signal together).
    """
    all_p, all_y, all_g = [], [], []
    for i in range(X.shape[0]):
        p = stft_log_profile(X[i], fs=fs, n_fft=n_fft, hop=hop)
        all_p.append(p)
        all_y.append(np.full(p.shape[0], y[i], dtype=np.int64))
        all_g.append(np.full(p.shape[0], groups[i], dtype=np.int64))
    return (
        np.concatenate(all_p, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_g, axis=0),
    )


def spectral_statistics(profile: np.ndarray) -> np.ndarray:
    """Five fixed across-bin descriptors per frame.

    profile : (n_frames, n_bins) log-magnitude (already energy-normalised).

    Returns
    -------
    stats : (n_frames, 5) — [flatness, entropy, centroid, skewness, kurtosis]
    These are computed on the *linear-magnitude* profile derived from
    ``exp(profile)`` since flatness/entropy are defined on PSD-positive
    quantities; skewness/kurtosis are computed on the log-profile directly
    (which is the across-bin sample whose moments the spec asks about).
    """
    n_frames, n_bins = profile.shape
    mag2 = np.exp(profile)  # back to (relative) magnitude-squared
    # Normalise so each frame's mag2 sums to 1 — turns it into a discrete pmf
    # over bins (needed for entropy and flatness consistently).
    s = mag2.sum(axis=1, keepdims=True)
    pmf = mag2 / np.where(s > 0, s, 1.0)
    # Flatness: geomean / arithmean of mag2.
    log_mag2 = np.log(mag2 + _LOG_EPS)
    geo = np.exp(log_mag2.mean(axis=1))
    ari = mag2.mean(axis=1)
    flatness = geo / np.where(ari > 0, ari, 1.0)
    # Entropy of the pmf, normalised to [0, 1] by log(n_bins).
    entropy = -np.sum(pmf * np.log(pmf + _LOG_EPS), axis=1) / np.log(n_bins)
    # Centroid (bin index, normalised to [0, 1]).
    bin_idx = np.arange(n_bins, dtype=np.float64) / max(n_bins - 1, 1)
    centroid = (pmf * bin_idx[None, :]).sum(axis=1)
    # Skew & kurtosis of the *log-profile* across bins (the moments named in §3).
    mu = profile.mean(axis=1, keepdims=True)
    d = profile - mu
    var = (d ** 2).mean(axis=1)
    std = np.sqrt(np.where(var > 0, var, 1e-12))
    skew = (d ** 3).mean(axis=1) / std ** 3
    kurt = (d ** 4).mean(axis=1) / std ** 4 - 3.0  # excess kurtosis
    return np.stack([flatness, entropy, centroid, skew, kurt], axis=1)


def class_moment_summary(
    profile: np.ndarray, y_frame: np.ndarray
) -> dict[int, dict]:
    """Per-class mean ± std of each across-bin moment — for H1 reporting."""
    stats = spectral_statistics(profile)
    names = ["flatness", "entropy", "centroid", "skewness", "kurtosis"]
    out: dict[int, dict] = {}
    for c in np.unique(y_frame):
        mask = y_frame == c
        per_stat = {}
        for j, nm in enumerate(names):
            v = stats[mask, j]
            per_stat[nm] = dict(mean=float(v.mean()), std=float(v.std()))
        out[int(c)] = per_stat
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)
    p = stft_log_profile(x, fs=20e6)
    print("profile:", p.shape, "stats:", spectral_statistics(p).shape)
