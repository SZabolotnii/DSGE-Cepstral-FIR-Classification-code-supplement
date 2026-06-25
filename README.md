# DSGE-Cepstral-FIR-Classification-code-supplement

Reproduction package for the paper

> **Cepstral FIR-Robust Classification: A Homomorphic-Deconvolution Descriptor
> in Space with a Generating Element** — S. V. Zabolotnii, 2026 (preprint, under
> review at the EURASIP Journal on Advances in Signal Processing).

## What the paper shows

Convolution in time is **addition in the cepstrum**
(`log|E·H| = log|E| + log|H|`), so an unknown convolutive channel (multipath,
FIR distortion) turns from a multiplicative spectral tilt into an *additive,
non-Gaussian perturbation* — exactly the regime where per-class DSGE
reconstruction (decomposition in a **space with a generating element**;
`F_c·K_c = B_c`, log-MSED feature) is strongest. Low-quefrency liftering removes
the smooth channel band before a class-specific reconstructor is applied, with
the downstream classifier and the descriptor budget held fixed.

The claim is deliberately narrow — a **representation rule**, not a SOTA method:
under injected convolutive distortion the liftered-cepstral reconstruction
classifier degrades *less* than its spectral-profile counterpart at equal
feature budget; on clean data the ordering reverses.

### Naming: paper ↔ code

| Paper name | Code label | What it is |
|---|---|---|
| **DB-MSED** | `C2` | per-class log-Mahalanobis reconstruction (`D²_c`) over the descriptor basis + logistic head |
| cepstral / spectral branch | `cepstral_prereg` / `spectral` | same C2 head over liftered real cepstrum vs over the log-spectral profile |

## Headline results (all checked by `verify_article_numbers.py`)

- **CWRU bearings, under injected FIR** (15 leakage-safe repeated splits, identical
  11-descriptor budget): cepstral C2 `0.596` vs spectral C2 `0.503` macro-F1,
  mean gap **+0.093**, 15/15 split wins, one-sided Wilcoxon `p < 0.05`, 95%
  bootstrap CI `[0.087, 0.098]` (excludes 0). On **clean** CWRU the gap closes
  (`0.680` vs `0.677`) — the gain is conditional on the convolutive nuisance.
- **DroneRF scout** (E8): the same effect, larger margin (gap **+0.244**).
- A **stronger MFCC baseline** (svm `0.845` under FIR) is reported honestly as a
  representation that *confirms the cepstral principle*, not as a method the
  paper beats.

## Layout

```
experiments/   experiment drivers + feature/descriptor code (see table below)
toolkit/       vendored DSGE toolkit (basis, F·K=B solver, leakage-safe splits, bootstrap)
results/       pinned fir_gate JSON artifacts every paper number is built from
data/          datasets (not committed — see data/README.md)
verify_article_numbers.py   re-checks every headline number against results/*.json
```

Key entry points:

| Script | Produces | Paper artifact |
|---|---|---|
| `verify_article_numbers.py` | console PASS/FAIL | re-derives every headline number from the JSONs (no data) |
| `experiments/fig_fir_revision.py` | `results/fir_gate/figs_en/*` | manuscript figures (from pinned JSONs; no data) |
| `experiments/run_fir_cwru_cv.py` | `fir_cwru_cv.json`, `fir_cwru_cv_branch.json` | E0 CWRU repeated-split FIR robustness (headline) |
| `experiments/run_fir_cwru_baselines.py` | `fir_cwru_baselines.json` | E1/E2 matched-budget baselines (detrend, CMVN, RASTA, MFCC) |
| `experiments/run_fir_cwru_ablation.py` | `fir_cwru_ablation.json` | E-ablation: full cepstrum vs liftered |
| `experiments/run_fir_lifter_bandpass.py` | `fir_cwru_lifter_bandpass.json` | E3 bandpass-lifter variant |
| `experiments/run_fir_cwru_sensitivity.py` | `fir_cwru_sensitivity.json` | E5/E6 ridge / `cond(F_c)` sensitivity |
| `experiments/run_fir_cwru_recordlevel.py` | `fir_cwru_recordlevel.json` | E7 recording-level aggregation |
| `experiments/run_fir_rf_cv.py` | `fir_rf_cv.json` | E8 DroneRF repeated-split |

## Quick start

```bash
pip install -r requirements.txt

# 1) Re-check every paper number against the pinned JSONs (no datasets needed):
python3 verify_article_numbers.py

# 2) Regenerate the manuscript figures from the pinned JSONs (no datasets needed):
python3 experiments/fig_fir_revision.py

# 3) Full reruns from raw signals require the datasets (see data/README.md), e.g.:
python3 experiments/run_fir_cwru_cv.py        # CWRU under data/cwru/
SDR_DRON_RAW=/path/to/dronerf/raw \
  python3 experiments/run_fir_rf_cv.py        # DroneRF scout
```

## Research discipline

Global seed `2026`. Leakage-safe splits (frames from one recording/time-block
never cross train/test). Equal feature budget across compared representations.
Lifter fixed *a priori* on the RF scout (`lift_lo = 3`) and transferred to CWRU
without tuning. Every headline claim passes repeated-split CV + paired one-sided
Wilcoxon + R=2000 bootstrap with a CI that excludes zero (`toolkit/verify_dsge.py`).
Per-class conditioning `cond(F_c)` is reported (Tikhonov ridge `1e-2`).

## Citation

See `CITATION.cff`. Please cite the paper, not only the repository.

## License

MIT — see `LICENSE`.
