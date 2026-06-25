# Datasets (not committed)

The pinned `results/fir_gate/*.json` artifacts let you reproduce every paper
number and figure **without any dataset** (`verify_article_numbers.py` and
`experiments/fig_fir_revision.py`). Datasets are only needed to *re-run* the
experiments from raw signals. They are not redistributed here.

## CWRU bearing-fault vibration — headline domain (E0, E1–E7)

Case Western Reserve University Bearing Data Center, 12 kHz drive-end
accelerometer records.

- Source: https://engineering.case.edu/bearingdatacenter
- Place the `.mat` files under `data/cwru/` (the experiment scripts read from
  there; adjust the in-script `DATA` path if you store them elsewhere).
- The bearing → sensor mechanical transfer path is a genuine FIR filter, which
  is why this is the convolutive-distortion test domain. Injected FIR channels
  are applied to clean windows at evaluation time (train clean / test under FIR).

## DroneRF — RF scout domain (E8)

DroneRF, 2.4 GHz band, interleaved int16 IQ at fs = 120 MHz.

- Source: Zenodo record **4264467** (https://doi.org/10.5281/zenodo.4264467).
- Point the `SDR_DRON_RAW` environment variable at the directory of `*_2G*.bin`
  raw files, e.g. `export SDR_DRON_RAW=/path/to/dronerf/raw` (default
  `data/dronerf/raw`).

## Notes

- Global seed: **2026** everywhere.
- Splits are leakage-safe: frames from one recording/time-block never cross
  train/test (`toolkit/splits.py`).
- `experiments/run_drilling.py` is imported only for shared shape-descriptor
  helpers; its drilling dataset is **not** required to reproduce the paper.
