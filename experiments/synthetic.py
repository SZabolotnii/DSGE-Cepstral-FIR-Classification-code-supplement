"""Synthetic 4-class complex-baseband signal generator for DSGE-Spectral H1–H4.

Each class has a distinct *spectral shape* (across-bin amplitude profile) so the
across-bin distribution carries class information. Per-realisation phases,
optional carrier offset, optional toolkit-style impairments, and AWGN provide
realistic within-class variability without erasing the shape signature.

Public API
----------
- ``CLASS_NAMES`` — list of human-readable class labels.
- ``make_dataset(n_per_class, n_samples, fs, fc, snr_db, ...)`` — produces a
  dict ``{X, y, groups, fs, fc}`` with complex64 signals.
- ``apply_cfo(x, fs, df_hz)`` — time-domain CFO injection (matches the toolkit's
  ``impairments._apply_cfo`` formula).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


CLASS_NAMES = [
    "broadband_uniform",   # C0: flat-ish, low across-bin kurtosis
    "sparse_peaked",       # C1: few isolated tones, very high kurtosis
    "multimodal_cluster",  # C2: 2-3 amplitude clusters, bimodal across bins
    "log_decay_envelope",  # C3: monotone decay, strong positive skew
]
N_CLASSES = len(CLASS_NAMES)


@dataclass(frozen=True)
class ClassTemplate:
    """Per-class spectral shape parameters.

    ``draw(rng, n_bins)`` produces a per-realisation amplitude profile over
    ``n_bins`` frequency bins. Per-bin phase is random uniform on [0, 2π); the
    iFFT of ``amp * exp(j*phase)`` is the time-domain signal.
    """
    name: str
    cid: int
    base_amp_fn: callable           # rng, n_bins -> (n_bins,) amplitudes
    amp_jitter: float = 0.10        # multiplicative jitter on amplitudes

    def draw_amp(self, rng: np.random.Generator, n_bins: int) -> np.ndarray:
        amp = self.base_amp_fn(rng, n_bins)
        jit = rng.normal(1.0, self.amp_jitter, size=n_bins)
        return np.clip(amp * jit, 1e-6, None)


def _broadband_uniform(rng: np.random.Generator, n_bins: int) -> np.ndarray:
    return np.ones(n_bins, dtype=np.float64) * (0.8 + 0.4 * rng.random())


def _sparse_peaked(rng: np.random.Generator, n_bins: int) -> np.ndarray:
    amp = np.full(n_bins, 0.02, dtype=np.float64)
    n_peaks = int(rng.integers(3, 7))
    peak_bins = rng.choice(np.arange(2, n_bins - 2), size=n_peaks, replace=False)
    for b in peak_bins:
        h = 5.0 + 5.0 * rng.random()
        amp[b] = h
        amp[b - 1] = 0.3 * h
        amp[b + 1] = 0.3 * h
    return amp


def _multimodal_cluster(rng: np.random.Generator, n_bins: int) -> np.ndarray:
    amp = np.full(n_bins, 0.05, dtype=np.float64)
    n_clusters = int(rng.integers(2, 4))
    bin_grid = np.arange(n_bins, dtype=np.float64)
    for _ in range(n_clusters):
        center = float(rng.integers(8, n_bins - 8))
        width = float(rng.uniform(4.0, 10.0))
        height = float(rng.uniform(1.0, 2.5))
        amp += height * np.exp(-0.5 * ((bin_grid - center) / width) ** 2)
    return amp


def _log_decay_envelope(rng: np.random.Generator, n_bins: int) -> np.ndarray:
    bin_grid = np.arange(n_bins, dtype=np.float64)
    decay_rate = float(rng.uniform(0.04, 0.08))
    amp = np.exp(-decay_rate * bin_grid)
    amp *= rng.uniform(0.85, 1.15, size=n_bins)
    amp += 0.03
    return amp


CLASS_TEMPLATES = [
    ClassTemplate("broadband_uniform", 0, _broadband_uniform, amp_jitter=0.05),
    ClassTemplate("sparse_peaked", 1, _sparse_peaked, amp_jitter=0.15),
    ClassTemplate("multimodal_cluster", 2, _multimodal_cluster, amp_jitter=0.12),
    ClassTemplate("log_decay_envelope", 3, _log_decay_envelope, amp_jitter=0.10),
]


def apply_cfo(x: np.ndarray, fs: float, df_hz: float) -> np.ndarray:
    """Time-domain CFO injection — same form as toolkit's _apply_cfo.

    x : complex array (1D), fs : Hz, df_hz : carrier offset in Hz.
    """
    if df_hz == 0.0:
        return x
    t = np.arange(len(x)) / fs
    return (x * np.exp(2j * np.pi * df_hz * t)).astype(x.dtype, copy=False)


def make_fir_channel(n_taps: int, rng: np.random.Generator,
                     decay: float = 0.7) -> np.ndarray:
    """Random complex multipath FIR channel of `n_taps` taps.

    Tap 0 is a dominant direct path; later taps are decaying random multipath
    echoes. Normalised to unit L2 energy so it does not change overall SNR.
    `n_taps = 1` is (a scaled) identity — the clean / no-channel point.
    """
    if n_taps <= 1:
        return np.array([1.0 + 0j], dtype=np.complex128)
    k = np.arange(n_taps)
    amp = np.exp(-decay * k)
    taps = amp * (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps))
    taps[0] = amp[0] * (1.0 + 0j)  # dominant, phase-aligned direct path
    taps = taps / np.sqrt(np.sum(np.abs(taps) ** 2))
    return taps.astype(np.complex128)


def apply_fir_channel(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Convolutional (multipath) distortion: y = x * h, length preserved.

    In the spectrum |Y| = |X|·|H|; in the log-spectrum this is an additive
    smooth term log|H|; in the cepstrum it adds a low-quefrency component.
    """
    if taps.size <= 1:
        return (x * taps[0]).astype(x.dtype, copy=False)
    y = np.convolve(x, taps, mode="full")[: len(x)]
    return y.astype(x.dtype, copy=False)


def _make_one_signal(
    template: ClassTemplate,
    rng: np.random.Generator,
    n_samples: int,
    fs: float,
    snr_db: float,
    cfo_hz: float = 0.0,
) -> np.ndarray:
    """Generate a single complex-baseband realisation."""
    # Spectral domain: amp profile + random phases (n_samples bins for iFFT).
    amp = template.draw_amp(rng, n_samples)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    spec = amp * np.exp(1j * phase)
    x = np.fft.ifft(spec)
    # Normalise to unit RMS before noise so SNR is meaningful.
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    if rms > 0:
        x = x / rms
    if cfo_hz != 0.0:
        x = apply_cfo(x, fs, cfo_hz)
    # AWGN at target SNR.
    sigma = 10.0 ** (-snr_db / 20.0) / np.sqrt(2.0)
    noise = sigma * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
    return (x + noise).astype(np.complex64)


def make_dataset(
    n_per_class: int = 300,
    n_samples: int = 4096,
    fs: float = 20e6,
    fc: float = 5.8e9,
    snr_db: float = 20.0,
    seed: int = 2026,
    cfo_hz_per_signal: float | None = None,
) -> dict:
    """Generate a synthetic multi-class dataset.

    Parameters
    ----------
    n_per_class : int   — signals per class.
    n_samples   : int   — samples per signal.
    fs, fc      : float — sample rate / carrier (kept consistent w/ toolkit defaults).
    snr_db      : float — additive complex-Gaussian noise SNR in dB.
    seed        : int   — master RNG seed.
    cfo_hz_per_signal : float or None
        If not None, every signal gets a *fixed* CFO of this many Hz at generation
        time. Use ``None`` for clean baseline; the H3 sweep passes specific values
        to perturb the test set.

    Returns
    -------
    dict with:
        X        : (N, n_samples) complex64
        y        : (N,) int class labels in [0..N_CLASSES)
        groups   : (N,) int signal-id (1-1 with samples — each signal = own group)
        fs, fc   : floats
        n_per_class, snr_db, seed : echoed parameters
    """
    rng = np.random.default_rng(seed)
    total = N_CLASSES * n_per_class
    X = np.empty((total, n_samples), dtype=np.complex64)
    y = np.empty(total, dtype=np.int64)
    groups = np.arange(total, dtype=np.int64)  # signal_id == row index
    idx = 0
    for tmpl in CLASS_TEMPLATES:
        for _ in range(n_per_class):
            cfo = 0.0 if cfo_hz_per_signal is None else float(cfo_hz_per_signal)
            X[idx] = _make_one_signal(tmpl, rng, n_samples, fs, snr_db, cfo_hz=cfo)
            y[idx] = tmpl.cid
            idx += 1
    return dict(
        X=X, y=y, groups=groups, fs=fs, fc=fc,
        n_per_class=n_per_class, snr_db=snr_db, seed=seed,
    )


if __name__ == "__main__":
    ds = make_dataset(n_per_class=5, n_samples=512, seed=0)
    print("X shape:", ds["X"].shape, "y unique:", np.unique(ds["y"]))
    print("groups range:", ds["groups"].min(), "..", ds["groups"].max())
