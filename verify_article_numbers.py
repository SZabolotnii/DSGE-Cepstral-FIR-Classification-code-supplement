"""Verify every headline number in the paper against the machine-saved JSON
artifacts. Exits non-zero on any mismatch.

Usage:  python3 verify_article_numbers.py [path-to-results/fir_gate]
Default results dir is ./results/fir_gate next to this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
CANDIDATES = [
    _HERE / "results" / "fir_gate",
]
RES = next((p for p in ([Path(sys.argv[1])] if len(sys.argv) > 1 else []) + CANDIDATES if p.exists()), None)
if RES is None:
    sys.exit("could not locate results/fir_gate")

fails = []


def chk(name, got, want, tol=0.001):
    ok = abs(got - want) <= tol
    print(f"  [{'OK ' if ok else 'XX '}] {name}: got {got:.4f}, expect {want:.4f} (tol {tol})")
    if not ok:
        fails.append(name)


def load(fn):
    return json.loads((RES / fn).read_text())

print(f"results dir: {RES}\n")

# E0 headline (identical descriptors)
cv = load("fir_cwru_cv.json")
assert cv["config"]["desc_mode"] == "identical", "fir_cwru_cv.json must be DESC_MODE=identical"
v = cv["verdict"]; a = cv["aggregate"]
print("E0 headline (identical descriptors):")
chk("spectral clean", a["spectral"]["clean_mean"], 0.677)
chk("spectral under-FIR", a["spectral"]["under_fir_mean"], 0.503)
chk("cepstral clean", a["cepstral_prereg"]["clean_mean"], 0.680)
chk("cepstral under-FIR", a["cepstral_prereg"]["under_fir_mean"], 0.596)
chk("gap mean", v["gap_mean"], 0.093)
chk("CI lo", v["bootstrap_ci95"][0], 0.087, tol=0.002)
chk("CI hi", v["bootstrap_ci95"][1], 0.098, tol=0.002)
assert v["wilcoxon_p"] < 0.05 and v["passed"] and v["frac_wins"] == 1.0

# branch (native descriptors)
br = load("fir_cwru_cv_branch.json")
print("E0 native (branch) descriptors:")
chk("branch gap", br["verdict"]["gap_mean"], 0.116)

# E1/E2 baselines
b = load("fir_cwru_baselines.json"); m = b["methods"]
print("E1/E2 baselines:")
chk("detrend under-FIR", m["detrend_spectral"]["under_fir_mean"], 0.616, tol=0.003)
chk("cmvn under-FIR", m["cmvn_spectral"]["under_fir_mean"], 0.180, tol=0.005)
chk("rasta under-FIR", m["rasta_spectral"]["under_fir_mean"], 0.427, tol=0.005)
chk("mfcc_dbmsed under-FIR", m["mfcc_dbmsed"]["under_fir_mean"], 0.810, tol=0.005)
chk("mfcc_svm under-FIR", m["mfcc_svm"]["under_fir_mean"], 0.845, tol=0.005)
chk("mfcc_svm clean", m["mfcc_svm"]["clean_mean"], 0.984, tol=0.003)

# E3 bandpass lifter
lb = load("fir_cwru_lifter_bandpass.json")["lifters"]
print("E3 bandpass lifter:")
chk("bp_3_64 clean", lb["bp_3_64"]["clean_mean"], 0.692, tol=0.003)
chk("bp_3_64 under-FIR", lb["bp_3_64"]["under_fir_mean"], 0.615, tol=0.003)

# E6 cond(F)
s = load("fir_cwru_sensitivity.json")
print("E5/E6 sensitivity:")
assert s["all_signs_stable"], "DSP/ridge sign stability failed"
chk("cond(F) median @ridge1e-2", s["ridge_sensitivity"]["0.01"]["cond_F_median"], 221.6, tol=5.0)

# E7 record-level
rl = load("fir_cwru_recordlevel.json")
print("E7 recording-level:")
chk("window-prob cepstral", rl["levels"]["cepstral_prob"]["under_fir_mean"], 0.830, tol=0.004)
chk("window-prob spectral", rl["levels"]["spectral_prob"]["under_fir_mean"], 0.685, tol=0.004)
chk("window-prob gap", rl["window_prob_gap"]["gap_mean"], 0.145, tol=0.004)
assert rl["window_prob_gap"]["passed"]

# E8 RF
rf = load("fir_rf_cv.json")
print("E8 DroneRF repeated-split:")
chk("RF gap", rf["verdict"]["gap_mean"], 0.244, tol=0.004)
chk("RF cepstral under-FIR", rf["aggregate"]["cepstral"]["under_fir_mean"], 0.470, tol=0.004)
chk("RF spectral under-FIR", rf["aggregate"]["spectral"]["under_fir_mean"], 0.227, tol=0.004)
assert rf["verdict"]["passed"]

print()
if fails:
    sys.exit(f"FAILED: {len(fails)} mismatches -> {fails}")
print("ALL MANUSCRIPT NUMBERS VERIFIED AGAINST JSON ARTIFACTS.")
