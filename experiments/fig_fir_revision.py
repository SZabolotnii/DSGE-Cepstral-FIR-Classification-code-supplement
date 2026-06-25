"""Revision figures (English labels) for the cepstral FIR-robustness paper.

Reads (all from machine-saved JSON; no invented numbers):
  results/fir_gate/fir_cwru_cv.json        (E0 headline, DESC_MODE=identical)
  results/fir_gate/fir_cwru_baselines.json (E1/E2 matched-budget baselines)
  results/fir_gate/fir_rf_cv.json          (E8 DroneRF repeated-split)
Writes English PNG+PDF into results/fir_gate/figs_en/ (copied into the
revision manuscript figs/). Run AFTER the experiment chain.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
RES = _HERE.parent / "results" / "fir_gate"
FIGS = RES / "figs_en"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 140})


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def fig_tradeoff(cwru):
    lc = cwru["aggregate"]["lifter_curve"]
    lifters = sorted(int(k) for k in lc)
    clean = [lc[str(L)]["clean_mean"] for L in lifters]
    under = [lc[str(L)]["under_fir_mean"] for L in lifters]
    under_sd = [lc[str(L)]["under_fir_std"] for L in lifters]
    clean_sd = [lc[str(L)]["clean_std"] for L in lifters]
    spec = cwru["aggregate"]["spectral"]
    prereg = cwru["config"]["prereg_lift"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = np.arange(len(lifters))
    ax.errorbar(x, clean, yerr=clean_sd, marker="o", capsize=3, label="cepstrum, clean", color="#1f77b4")
    ax.errorbar(x, under, yerr=under_sd, marker="s", capsize=3, label="cepstrum, under FIR", color="#d62728")
    ax.axhline(spec["clean_mean"], ls="--", color="#1f77b4", alpha=0.6, label="spectral, clean")
    ax.axhline(spec["under_fir_mean"], ls="--", color="#d62728", alpha=0.6, label="spectral, under FIR")
    ip = lifters.index(prereg)
    ax.axvspan(ip - 0.3, ip + 0.3, color="gold", alpha=0.25)
    ax.annotate(f"pre-registered\n$\\ell$={prereg}", (ip, under[ip]), textcoords="offset points",
                xytext=(8, -34), fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xticks(x); ax.set_xticklabels([str(L) for L in lifters])
    ax.set_xlabel("lower-bound lifter $\\ell$ (low quefrencies discarded)")
    ax.set_ylabel("macro-$F_1$")
    ax.set_title("CWRU: clean-vs-robustness trade-off across lifters")
    ax.legend(fontsize=8, loc="lower center", ncol=2)
    _save(fig, "fig1_tradeoff_cwru")


def fig_severity(cwru):
    per = cwru["per_split"]; fir = cwru["config"]["fir_sweep"]; prereg = cwru["config"]["prereg_lift"]
    def avg(getter):
        M = np.array([[getter(s)[str(L)] for L in fir] for s in per]); return M.mean(0), M.std(0)
    sm, ss = avg(lambda s: s["spectral"]["sweep"])
    fm, fs = avg(lambda s: s["cepstral"]["1"]["sweep"])
    lm, ls = avg(lambda s: s["cepstral"][str(prereg)]["sweep"])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = np.arange(len(fir))
    ax.errorbar(x, sm, yerr=ss, marker="o", capsize=3, label="spectral (DB-MSED)")
    ax.errorbar(x, fm, yerr=fs, marker="^", capsize=3, label="full cepstrum (DB-MSED)")
    ax.errorbar(x, lm, yerr=ls, marker="s", capsize=3, color="#2ca02c",
                label=f"cepstrum lifter $\\ell$={prereg} (DB-MSED)")
    ax.set_xticks(x); ax.set_xticklabels([("clean" if L == 1 else str(L)) for L in fir])
    ax.set_xlabel("FIR channel length (taps)"); ax.set_ylabel("macro-$F_1$")
    ax.set_title("CWRU: degradation under convolutional distortion")
    ax.legend(fontsize=9)
    _save(fig, "fig2_severity_cwru")


def fig_gap(cwru):
    per = cwru["per_split"]; gaps = np.array([s["gap_prereg"] for s in per])
    v = cwru["verdict"]; ci = v["bootstrap_ci95"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(gaps))
    ax.bar(x, gaps, color=np.where(gaps > 0, "#2ca02c", "#d62728"), alpha=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(gaps.mean(), color="navy", lw=1.5, label=f"mean = {gaps.mean():+.3f}")
    ax.axhspan(ci[0], ci[1], color="navy", alpha=0.15, label=f"95% bootstrap CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    ax.set_xlabel("split index (leakage-safe, 15 repeats)")
    ax.set_ylabel("$\\Delta$ macro-$F_1$ under FIR (cepstrum $\\ell$3 $-$ spectral)")
    ax.set_title(f"CWRU: cepstral advantage under FIR on {int((gaps>0).sum())}/{len(gaps)} splits, "
                 f"Wilcoxon $p$={v['wilcoxon_p']:.1e}")
    ax.legend(fontsize=9)
    _save(fig, "fig3_gap_cwru")


def fig_cross_domain(cwru, rf):
    cw_gap = cwru["verdict"]["gap_mean"]; cw_ci = cwru["verdict"]["bootstrap_ci95"]
    rf_gap = rf["verdict"]["gap_mean"]; rf_ci = rf["verdict"]["bootstrap_ci95"]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    labels = ["DroneRF\n(15 splits)", "CWRU\n(15 splits)"]
    vals = [rf_gap, cw_gap]
    errs = [[rf_gap - rf_ci[0], cw_gap - cw_ci[0]], [rf_ci[1] - rf_gap, cw_ci[1] - cw_gap]]
    bars = ax.bar(labels, vals, color=["#9467bd", "#2ca02c"], alpha=0.85, yerr=errs, capsize=5)
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.003, f"{val:+.3f}", ha="center", fontsize=10)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("$\\Delta$ macro-$F_1$ under FIR (cepstrum $-$ spectral)")
    ax.set_title("Cross-domain FIR robustness (repeated splits)")
    _save(fig, "fig4_cross_domain")


def fig_baselines(cwru, base):
    spec = cwru["aggregate"]["spectral"]
    rows = [("spectral", spec["under_fir_mean"], spec["clean_mean"])]
    order = ["detrend_spectral", "rasta_spectral", "cmvn_spectral"]
    names = {"detrend_spectral": "+ detrend", "rasta_spectral": "+ RASTA", "cmvn_spectral": "+ CMVN"}
    for k in order:
        m = base["methods"][k]; rows.append((names[k], m["under_fir_mean"], m["clean_mean"]))
    rows.append(("cepstrum $\\ell$3", base["cepstral_l3"]["under_fir_mean"], base["cepstral_l3"]["clean_mean"]))
    for k, lab in [("mfcc_dbmsed", "MFCC$\\to$DB-MSED"), ("mfcc_svm", "MFCC$\\to$SVM"), ("mfcc_gmm", "MFCC$\\to$GMM")]:
        m = base["methods"][k]; rows.append((lab, m["under_fir_mean"], m["clean_mean"]))
    labels = [r[0] for r in rows]; under = [r[1] for r in rows]; clean = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(labels))
    ax.barh(y - 0.2, clean, height=0.4, label="clean", color="#1f77b4", alpha=0.85)
    ax.barh(y + 0.2, under, height=0.4, label="under FIR", color="#d62728", alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("macro-$F_1$"); ax.set_xlim(0, 1.0)
    ax.set_title("CWRU: matched 11-feature budget baselines")
    ax.legend(fontsize=9, loc="lower right")
    _save(fig, "fig5_baselines")


def main():
    cwru = json.loads((RES / "fir_cwru_cv.json").read_text())
    print("Revision figures (English):")
    fig_tradeoff(cwru); fig_severity(cwru); fig_gap(cwru)
    bp = RES / "fir_cwru_baselines.json"
    if bp.exists():
        fig_baselines(cwru, json.loads(bp.read_text()))
    rfp = RES / "fir_rf_cv.json"
    if rfp.exists():
        fig_cross_domain(cwru, json.loads(rfp.read_text()))
    else:
        print("  (fir_rf_cv.json not yet present — skip fig4 cross-domain)")
    print(f"All figures in {FIGS}")


if __name__ == "__main__":
    main()
