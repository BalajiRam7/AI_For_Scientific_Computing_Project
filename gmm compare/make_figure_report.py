"""Figure 1 of the midway report: one panel per experiment (E1 OU exact fields, E2 double well,
E3 Gaussian-mixture jump-diffusion) plus the E3 spurious inter-mode mass. Writes ../fig_prelim.pdf."""
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "lines.linewidth": 1.1,
                     "lines.markersize": 2.6, "axes.linewidth": 0.5, "mathtext.fontset": "stix"})
STY = {"SI-SDE": ("SI-SDE", "#1f5fa8", "-", "o"), "SI-Follmer": ("SI-F\u00f6llmer", "#5fb0ef", "-", "s"),
       "Diff-SDE": ("Diff-SDE", "#c0392b", "--", "v"), "Diff-DDIM": ("Diff-DDIM", "#e8913a", "--", "^"),
       "Diff-SDE-resid": ("Diff-SDE-resid", "#1e8449", ":", "D"), "Diff-DDIM-resid": ("Diff-DDIM-resid", "#82c46c", ":", "x")}


def band(ax, x, v, c, ls, mk, lab=None):
    mu, sd = v.mean(0), v.std(0)
    ax.plot(x, mu, color=c, ls=ls, marker=mk, label=lab)
    if len(v) > 1:
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.18, lw=0)


fig, axs = plt.subplots(1, 4, figsize=(5.5, 1.36), constrained_layout=True)
# (a) E1: OU, exact fields, tau = 0.05
ou = json.load(open("results/results_ou_exact.json")); r = ou["runs"][0]
for k, (lab, c, ls, mk) in STY.items():
    key = {"SI-SDE": "SI-SDE(eps=1.0)", "SI-Follmer": "SI-Follmer(eps=1.0)"}.get(k, k)
    axs[0].plot(ou["Ks"], r[key], color=c, ls=ls, marker=mk, label=lab)
axs[0].axhline(r["floor"], color="0.6", lw=0.7, zorder=0)
axs[0].set_title("(a) E1: OU, $\\tau=0.05$\nexact fields"); axs[0].set_ylabel("relative $W_2$")
# (b) E2: double well, tau = 0.1, learned
runs = [json.load(open(f)) for f in sorted(glob.glob("results/results_dw_tau0.1_seed*.json"))]
for k, (lab, c, ls, mk) in STY.items():
    band(axs[1], runs[0]["Ks"], np.array([[np.mean(x) for x in q[k]] for q in runs]), c, ls, mk)
axs[1].axhline(np.mean([np.mean(q["floor"]) for q in runs]), color="0.6", lw=0.7, zorder=0)
axs[1].set_title("(b) E2: double well\n$\\tau=0.1$, learned"); axs[1].set_ylabel("$W_1$")
# (c) E3: GMM jump-diffusion, tau = 1, learned
post = json.load(open("results/gmm_posthoc.json"))
runs = [json.load(open(f)) for f in sorted(glob.glob("results/results_gmm_tau1.0_seed*.json"))]
for k, (lab, c, ls, mk) in STY.items():
    band(axs[2], runs[0]["Ks"], np.array([np.array(q[k])[:, :, 0].mean(1) for q in runs]), c, ls, mk)
axs[2].axhline(np.mean(np.array(post["1.0"]["floor_mean"])[:, 0]), color="0.6", lw=0.7, zorder=0)
axs[2].set_title("(c) E3: GMM jumps\n$\\tau=1$, learned"); axs[2].set_ylabel("sliced $W_1$")
for ax in axs[:3]:
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("NFE $K$"); ax.grid(True, lw=0.3, alpha=0.5)
axs[2].set_yticks([0.25, 0.4, 0.8]); axs[2].set_yticklabels(["0.25", "0.4", "0.8"]); axs[2].minorticks_off()
# (d) E3: fraction of forecasts between modes at K = 100, vs lag
taus = [0.5, 1.0, 2.0]
for k, (lab, c, ls, mk) in STY.items():
    v = np.array([np.array(post[str(t)]["offmode"][k]).mean(1) for t in taus]).T   # seeds x tau
    band(axs[3], taus, v, c, ls, mk)
axs[3].plot(taus, [np.mean(post[str(t)]["offmode_truth"]) for t in taus], color="k", ls="-.", lw=1.0,
            marker="*", ms=4, label="exact forecasts")
axs[3].set_yscale("log"); axs[3].set_xticks(taus); axs[3].set_xticklabels(["0.5", "1", "2"])
axs[3].set_xlabel("lag $\\tau$"); axs[3].set_ylabel("off-mode fraction"); axs[3].grid(True, lw=0.3, alpha=0.5)
axs[3].set_title("(d) E3: off-mode\nmass, $K=100$")
h, l = axs[0].get_legend_handles_labels(); h2, l2 = axs[3].get_legend_handles_labels()
fig.legend(h + h2, l + l2, loc="outside upper center", ncol=7, frameon=False, handlelength=1.7, columnspacing=0.7, fontsize=5.6)
fig.savefig("../fig_prelim.pdf")
print("saved")
