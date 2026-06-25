"""Real-data builder: DroneRF (Zenodo 4264467) 2G band → frame-level dataset.

Source: ``DroneRF 2G-band *_2G*.bin files (see data/README.md)`` —
interleaved int16 IQ at fs = 120 MHz, fc = 2.44 GHz.  Manifest:
``data/metadata/zenodo_4264467_manifest.csv``.

We pick the 2G band only so fs/fc/n_fft choices stay uniform across classes,
giving 9 distinct drone-model classes (Yuneec is split across two files
``..._1of2`` / ``..._2of2`` that we merge as one group of time-blocks;
Parrot mambo control & video are kept as separate classes because they
are functionally different transmitters from the same airframe).

Per file we read ``n_windows`` non-overlapping windows of ``win_samples``
complex samples, randomly spaced across the recording. Each window
gives ~``win_samples / hop`` STFT frames. To exclude near-silence frames
(the DroneRF recording captures the full 120 MHz band but each emitter
is narrowband, so most frames are noise floor only), we filter frames
by per-frame energy: a frame is kept iff its post-normalisation log
profile has a standard deviation above ``energy_std_thresh`` (a tiny
profile flat in log domain = noise floor; a bursty real emission has
sharp peaks → high std).

Leakage-safe groups: ``(file_basename, time_block_id)`` where
``time_block_id`` is the chunk of the recording the window came from.
``grouped_split`` keeps all frames from one time-block together — no
single-window or single-burst can straddle train/test.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# ----------------------------- File discovery ----------------------------- #

_RAW_DIR = Path(
    os.environ.get(
        "SDR_DRON_RAW", "data/dronerf/raw"
    )
)


def group_id_for(rec_idx: int, block_idx: int, n_time_blocks: int) -> int:
    """Deterministic, process-stable group id for (recording, time-block).

    ``rec_idx`` is the recording's index in the sorted ``discover_2g_recordings``
    output, so the same physical block always maps to the same id across runs
    (unlike ``hash()``, which is salted by PYTHONHASHSEED).
    """
    return rec_idx * n_time_blocks + block_idx


def _parse_filename(name: str) -> tuple[str, str, int | None, int | None]:
    """Return ``(vendor, model_slug, part, total_parts)``. Parts come from
    suffixes like ``_1of2``."""
    base = re.sub(r"\.bin$", "", name)
    m = re.search(r"_(\d+)of(\d+)$", base)
    part = total = None
    if m:
        part = int(m.group(1))
        total = int(m.group(2))
        base = base[: m.start()]
    parts = base.split("_")
    vendor = parts[0]
    band = parts[-1]
    if band not in {"2G", "5G"}:
        raise ValueError(f"Unexpected band suffix in {name!r}")
    role = None
    # Parrot_mambo_control_2G or Parrot_mambo_video_2G — preserve role.
    if vendor == "Parrot" and len(parts) >= 4 and parts[-2] in {"control", "video"}:
        role = parts[-2]
        model = "_".join(parts[1:-2])
    else:
        model = "_".join(parts[1:-1])
    model_slug = f"{vendor}_{model}" + (f"_{role}" if role else "")
    return model_slug, band, part, total


@dataclass(frozen=True)
class RecordingMeta:
    path: Path
    class_label: str
    fs: float
    fc: float
    n_complex: int
    part: int | None
    total_parts: int | None

    @property
    def class_group(self) -> str:
        """Same recording across multi-part files maps to one class label."""
        return self.class_label


# Per-band acquisition parameters (Zenodo 4264467 manifest).
_BAND_PARAMS = {
    "2G": dict(fs=120e6, fc=2.44e9),   # 1.0 s recordings → 120e6 complex samples
    "5G": dict(fs=200e6, fc=5.8e9),    # 0.5 s recordings → 100e6 complex samples
}


def discover_recordings(band: str = "2G", raw_dir: Path = _RAW_DIR) -> list[RecordingMeta]:
    """Find all recordings of a band under raw_dir and return parsed metadata.

    fs/fc come from the Zenodo manifest per band. Multi-part files
    (``_1of2`` / ``_2of2``) keep the same ``class_label`` and so merge into one
    drone class.
    """
    if band not in _BAND_PARAMS:
        raise ValueError(f"Unknown band {band!r}; choose from {list(_BAND_PARAMS)}")
    fs = _BAND_PARAMS[band]["fs"]
    fc = _BAND_PARAMS[band]["fc"]
    bytes_per_complex_int16 = 4
    out: list[RecordingMeta] = []
    for p in sorted(raw_dir.glob(f"*_{band}*.bin")):
        label, b, part, total = _parse_filename(p.name)
        sz = p.stat().st_size
        if sz % bytes_per_complex_int16 != 0:
            continue
        n_complex = sz // bytes_per_complex_int16
        out.append(RecordingMeta(
            path=p, class_label=label, fs=fs, fc=fc,
            n_complex=n_complex, part=part, total_parts=total,
        ))
    return out


def discover_2g_recordings(raw_dir: Path = _RAW_DIR) -> list[RecordingMeta]:
    """Back-compat wrapper — 2G band."""
    return discover_recordings("2G", raw_dir=raw_dir)


# ----------------------------- IO ----------------------------- #

def read_iq_window(path: Path, start_sample: int, sample_count: int) -> np.ndarray:
    """Read complex-baseband window from an interleaved int16 little-endian
    IQ file.  Normalised by 2¹⁵ so values land in [−1, 1]."""
    if start_sample < 0:
        raise ValueError("start_sample must be ≥ 0")
    if sample_count <= 0:
        raise ValueError("sample_count must be > 0")
    raw = np.memmap(
        str(path), dtype="<i2", mode="r",
        offset=start_sample * 4,
        shape=(sample_count * 2,),
    )
    paired = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
    cx = (paired[:, 0] + 1j * paired[:, 1]).astype(np.complex64) / np.float32(32768.0)
    return cx


# ----------------------------- Dataset assembly ----------------------------- #

def _energy_std(log_profile: np.ndarray) -> np.ndarray:
    """Std-dev of each log-profile frame across bins — a bursty emission
    has sharp peaks → high std; pure-noise frame is flat → low std."""
    return log_profile.std(axis=1)


def build_real_dataset(
    n_windows_per_file: int = 200,
    win_samples: int = 4096,
    n_fft: int = 256,
    hop: int = 128,
    n_time_blocks: int = 20,
    energy_std_thresh: float = 0.8,
    max_frames_per_class: int | None = None,
    seed: int = 2026,
    recordings: list[RecordingMeta] | None = None,
    raw_dir: Path = _RAW_DIR,
) -> dict:
    """Return frame-level dataset dict identical in shape to the synthetic
    one:

    Returns
    -------
    dict with:
        profiles : (N_frames, n_bins) log-magnitude STFT frame, energy-
                   normalised per-frame (DC bin dropped → n_bins = n_fft − 1).
        y        : (N_frames,) int class label.
        groups   : (N_frames,) int group id = (class, time_block).
        meta     : list of dicts, one per recording, with frame counts.
        class_names : list of class label strings.
        fs, n_fft, hop : echoed.

    Each .bin file is split into ``n_time_blocks`` equal time blocks. From each
    block we draw ``n_windows_per_block`` random windows of ``win_samples``
    samples, STFT them, energy-filter by per-frame log-magnitude std, and
    accumulate frames. The block-id is part of the group label so a grouped
    split keeps frames from one block on one side.
    """
    if recordings is None:
        recordings = discover_2g_recordings(raw_dir=raw_dir)
    rng = np.random.default_rng(seed)

    # Build label index: each unique class_label → int id.
    label_to_id: dict[str, int] = {}
    for r in recordings:
        if r.class_label not in label_to_id:
            label_to_id[r.class_label] = len(label_to_id)
    class_names = [name for name, _ in sorted(label_to_id.items(), key=lambda kv: kv[1])]
    n_classes = len(class_names)

    # ``time_block`` is per *recording-file*; we then concatenate with
    # ``class_id`` to produce a globally-unique group id so grouped_split
    # never crosses class lines while still keeping block-level locality.
    n_windows_per_block = max(1, n_windows_per_file // n_time_blocks)

    profiles_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    groups_all: list[np.ndarray] = []
    meta_rows: list[dict] = []
    per_class_kept = {c: 0 for c in range(n_classes)}

    # Import the STFT routine from features.py.
    from features import stft_log_profile

    for rec_idx, rec in enumerate(recordings):
        cid = label_to_id[rec.class_label]
        if max_frames_per_class is not None and per_class_kept[cid] >= max_frames_per_class:
            meta_rows.append(dict(
                file=rec.path.name, class_label=rec.class_label, class_id=cid,
                kept_frames=0, drawn_windows=0, skipped_reason="cap reached",
            ))
            continue

        block_size = rec.n_complex // n_time_blocks
        if block_size < win_samples:
            # Should not happen for 1-second @120MHz; safety check.
            continue

        kept_here = 0
        drawn_here = 0
        for block_idx in range(n_time_blocks):
            block_start = block_idx * block_size
            block_end = (block_idx + 1) * block_size - win_samples
            if block_end <= block_start:
                continue
            # Deterministic globally-unique group id (file × block). Uses the
            # recording's stable index in the sorted discovery order — NOT
            # Python hash(), which is PYTHONHASHSEED-salted and would make the
            # train/test split non-reproducible across processes.
            group_id = group_id_for(rec_idx, block_idx, n_time_blocks)
            for _ in range(n_windows_per_block):
                start = int(rng.integers(block_start, block_end + 1))
                drawn_here += 1
                x = read_iq_window(rec.path, start_sample=start,
                                    sample_count=win_samples)
                p = stft_log_profile(x, fs=rec.fs, n_fft=n_fft, hop=hop)
                mask = _energy_std(p) >= energy_std_thresh
                if not np.any(mask):
                    continue
                p_kept = p[mask]
                profiles_all.append(p_kept)
                y_all.append(np.full(p_kept.shape[0], cid, dtype=np.int64))
                groups_all.append(np.full(p_kept.shape[0], group_id, dtype=np.int64))
                kept_here += int(p_kept.shape[0])
                per_class_kept[cid] += int(p_kept.shape[0])
                if (max_frames_per_class is not None
                        and per_class_kept[cid] >= max_frames_per_class):
                    break
            if (max_frames_per_class is not None
                    and per_class_kept[cid] >= max_frames_per_class):
                break

        meta_rows.append(dict(
            file=rec.path.name, class_label=rec.class_label, class_id=cid,
            kept_frames=kept_here, drawn_windows=drawn_here,
            n_complex=rec.n_complex,
        ))

    if not profiles_all:
        raise RuntimeError("No frames survived the energy filter — lower energy_std_thresh.")

    profiles = np.concatenate(profiles_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    groups = np.concatenate(groups_all, axis=0)

    return dict(
        profiles=profiles, y=y, groups=groups,
        class_names=class_names,
        n_classes=n_classes,
        fs=recordings[0].fs if recordings else 120e6,
        n_fft=n_fft, hop=hop,
        meta=meta_rows,
        seed=seed,
        n_windows_per_file=n_windows_per_file,
        win_samples=win_samples,
        n_time_blocks=n_time_blocks,
        energy_std_thresh=energy_std_thresh,
        per_class_kept=per_class_kept,
    )


if __name__ == "__main__":
    recs = discover_2g_recordings()
    print(f"Discovered {len(recs)} recordings:")
    for r in recs:
        print(f"  {r.class_label:<35} {r.path.name:<35} "
              f"n_complex={r.n_complex:,}")
    print(f"\nUnique class labels: {sorted({r.class_label for r in recs})}")
