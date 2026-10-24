"""Figures for Experiment 3 (GMM jump-diffusion): forecast samples / angular densities and error vs NFE."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exp_gmm_jump import MEANS

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2, "lines.markersize": 3,
                     "mathtext.fontset": "stix"})
STY = {"SI-SDE": ("SI-SDE", "#1f5fa8", "-", "o"), "SI-Follmer": ("SI-F\u00f6llmer", "#5fb0ef", "-", "s"),
       "Diff-SDE": ("Diff-SDE", "#c0392b", "--", "v"), "Diff-DDIM": ("Diff-DDIM", "#e8913a", "--", "^"),
       "Diff-SDE-resid": ("Diff-SDE-resid", "#1e8449", ":", "D"), "Diff-DDIM-resid": ("Diff-DDIM-resid", "#82c46c", ":", "x")}
TAUS = [0.5, 1.0, 2.0]
import glob
RUNS = {t: [json.load(open(f)) for f in sorted(glob.glob(f"results/results_gmm_tau{t}_seed*.json"))] for t in TAUS}
R = {t: RUNS[t][0] for t in TAUS}
POST = json.load(open("results/gmm_posthoc.json"))
NSEED = min(len(v) for v in RUNS.values())


def stat(tau, key, metric):
    """(seeds, K) array of a metric averaged over the four test x0."""
    return np.array([np.array(r_[key])[:, :, metric].mean(1) for r_ in RUNS[tau]])

# ---------- Figure A: samples from x0 = mode centre at K = 100, and angular densities
cols = ["Truth", "SI-Follmer", "Diff-SDE", "Diff-DDIM", "Diff-SDE-resid"]
fig, axs = plt.subplots(3, 6, figsize=(12, 6.2), constrained_layout=True)
bins = np.linspace(-np.pi, np.pi, 91); mid = 0.5 * (bins[1:] + bins[:-1])
for r, tau in enumerate(TAUS):
    gt = np.load(f"results/gt_gmm_tau{tau}.npz")["gt"][0]
    smp = np.load(f"results/samples_gmm_tau{tau}_seed0.npz")
    x0 = np.array(R[tau]["x0_test"][0])
    for c, name in enumerate(cols):
        ax = axs[r, c]
        xs = gt[:2000] if name == "Truth" else smp[name][0]
        ax.scatter(xs[:, 0], xs[:, 1], s=0.6, alpha=0.35, color="0.25" if name == "Truth" else STY[name][1], rasterized=True)
        ax.scatter(MEANS[:, 0], MEANS[:, 1], marker="+", color="0.5", s=25, lw=0.8)
        ax.scatter(*x0, marker="*", color="k", s=40, zorder=5)
        ax.set_xlim(-9, 9); ax.set_ylim(-9, 9); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(name.replace("Follmer", "F\u00f6llmer") + ("" if name == "Truth" else " (K=100)"))
        if c == 0: ax.set_ylabel(f"$\\tau={tau}$", fontsize=9)
    ax = axs[r, 5]
    sm = lambda h: np.convolve(h, np.ones(3) / 3, mode="same")
    ax.fill_between(mid, sm(np.histogram(np.arctan2(gt[:, 1], gt[:, 0]), bins, density=True)[0]), color="0.8", label="Truth")
    for name in cols[1:]:
        th = np.arctan2(smp[name][0][:, 1], smp[name][0][:, 0])
        h = sm(np.histogram(th, bins, density=True)[0])
        ax.plot(mid, h, color=STY[name][1], ls=STY[name][2], lw=1.0, label=STY[name][0])
    ax.set_xlim(-np.pi, np.pi); ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["$-\\pi$", "0", "$\\pi$"])
    if r == 0: ax.set_title("angular density"); ax.legend(frameon=False, fontsize=6)
fig.suptitle("Forecasts from $x_0=(5,0)$ (black star); gray crosses mark the five mode centres", fontsize=9)
fig.savefig("../fig_gmm_samples.png", dpi=170)

# ---------- Figure B: error vs NFE (mean over the four test x0; band = +-1 s.d. over seeds) + spurious mass
fig, axs = plt.subplots(3, 3, figsize=(9, 6.6), constrained_layout=True)
for c, tau in enumerate(TAUS):
    Ks = R[tau]["Ks"]
    for row, lab in enumerate(["sliced $W_1$", "mode-occupation TV"]):
        ax = axs[row, c]
        for k, (nm, col, ls, mk) in STY.items():
            v = stat(tau, k, row); mu, sd = v.mean(0), v.std(0)
            ax.plot(Ks, mu, color=col, ls=ls, marker=mk, label=nm)
            if len(v) > 1: ax.fill_between(Ks, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        ax.axhline(np.mean(np.array(POST[str(tau)]["floor_mean"])[:, row]), color="0.6", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, lw=0.3, alpha=0.5)
        if row == 0: ax.set_title(f"$\\tau={tau}$")
        if row == 1: ax.set_xlabel("NFE $K$")
        if c == 0: ax.set_ylabel(lab)
    ax = axs[2, c]
    om = POST[str(tau)]["offmode"]
    names = list(STY)
    vals = [np.mean(om[k]) for k in names]; errs = [np.std(np.mean(om[k], 1)) for k in names]
    ax.bar(range(len(names)), vals, yerr=errs, color=[STY[k][1] for k in names], capsize=2)
    ax.axhline(np.mean(POST[str(tau)]["offmode_truth"]), color="k", lw=0.9, ls="--", label="exact forecasts")
    ax.set_xticks(range(len(names))); ax.set_xticklabels([STY[k][0] for k in names], rotation=35, ha="right", fontsize=6.5)
    if c == 0: ax.set_ylabel("off-mode fraction ($K=100$)"); ax.legend(frameon=False, fontsize=6.5)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=6, frameon=False)
fig.savefig("../fig_gmm_metrics.pdf"); fig.savefig("../fig_gmm_metrics.png", dpi=170)

# ---------- console tables
for tau in TAUS:
    r_ = R[tau]; Ks = r_["Ks"]
    fm = np.array(POST[str(tau)]["floor_mean"]).mean(0)
    print(f"\n tau={tau}  ({len(RUNS[tau])} seeds)  expected floor SW1={fm[0]:.3f}  modeTV={fm[1]:.3f}")
    print("   mean over x0, SW1 at K=" + str(Ks) + "   |  modeTV")
    for k in STY:
        a0, a1 = stat(tau, k, 0), stat(tau, k, 1)
        print(f"   {k:16s}" + " ".join(f"{m:.3f}+-{d:.3f}" for m, d in zip(a0.mean(0), a0.std(0))) + "  | " + " ".join(f"{m:.3f}" for m in a1.mean(0)))
    j = Ks.index(100)
    print("   per-x0 SW1 at K=100:", ", ".join(r_["x0_names"]))
    for k in STY:
        print(f"   {k:16s}" + " ".join(f"{np.mean([np.array(q[k])[j, i, 0] for q in RUNS[tau]]):6.3f}" for i in range(4)))
    print(f"   {'floor':16s}" + " ".join(f"{v:6.3f}" for v in np.array(POST[str(tau)]['floor_mean'])[:, 0]))
    om = POST[str(tau)]["offmode"]
    print("   off-mode fraction at K=100 (truth %.4f): " % np.mean(POST[str(tau)]["offmode_truth"]) + ", ".join(f"{k} {np.mean(v):.4f}" for k, v in om.items()))
print("saved figures")
