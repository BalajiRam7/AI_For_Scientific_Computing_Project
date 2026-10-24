"""Builds Figure 1 of the midway report from results_ou_exact.json and results_dw_*.json."""
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "lines.linewidth": 1.1, "lines.markersize": 2.6,
                     "axes.linewidth": 0.5, "mathtext.fontset": "stix"})
STY = {"SI-SDE": ("SI-SDE ($g=\\sigma$)", "#1f5fa8", "-", "o"),
       "SI-Follmer": ("SI-F\u00f6llmer ($g=g^{\\rm F}$)", "#5fb0ef", "-", "s"),
       "Diff-SDE": ("Diff-SDE", "#c0392b", "--", "v"),
       "Diff-DDIM": ("Diff-DDIM", "#e8913a", "--", "^"),
       "Diff-SDE-resid": ("Diff-SDE-resid", "#1e8449", ":", "D"),
       "Diff-DDIM-resid": ("Diff-DDIM-resid", "#82c46c", ":", "x")}

ou = json.load(open("results/results_ou_exact.json"))
KS = ou["Ks"]
fig, axs = plt.subplots(1, 4, figsize=(5.5, 1.6), constrained_layout=True)
for ax, r, ttl in zip(axs[:2], [ou["runs"][0], ou["runs"][2]], ["(a) OU, $\\tau=0.05$\nexact drift/score",
                                                                  "(b) OU, $\\tau=1$\nexact drift/score"]):
    for k, (lab, c, ls, mk) in STY.items():
        key = k.replace("SI-SDE", "SI-SDE(eps=1.0)").replace("SI-Follmer", "SI-Follmer(eps=1.0)")
        ax.plot(KS, r[key], ls=ls, color=c, marker=mk, label=lab)
    ax.axhline(r["floor"], color="0.6", lw=0.7, zorder=0)
    ax.set_title(ttl); ax.set_ylabel("rel. $W_2$" if ax is axs[0] else "")

for ax, tau, ttl in zip(axs[2:], [0.1, 2.0], ["(c) Double well, $\\tau=0.1$\nlearned",
                                             "(d) Double well, $\\tau=2$\nlearned"]):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"results/results_dw_tau{tau}_seed*.json"))]
    Kd = runs[0]["Ks"]
    for k, (lab, c, ls, mk) in STY.items():
        vals = np.array([[np.mean(x) for x in r_[k]] for r_ in runs])   # seeds x K
        mu, sd = vals.mean(0), vals.std(0)
        ax.plot(Kd, mu, ls=ls, color=c, marker=mk)
        ax.fill_between(Kd, mu - sd, mu + sd, color=c, alpha=0.18, lw=0)
    ax.axhline(np.mean([np.mean(r_["floor"]) for r_ in runs]), color="0.6", lw=0.7, zorder=0)
    ax.set_title(ttl + f", {len(runs)} seeds")
    ax.set_ylabel("$W_1$" if ax is axs[2] else ""); ax.set_ylim(5e-3, 1.2)
for ax in axs:
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("NFE $K$")
    ax.grid(True, which="major", lw=0.3, alpha=0.5)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=3, frameon=False, handlelength=2.4, columnspacing=1.5)
fig.savefig("../fig_e1e2_lags.pdf")
print("saved")

# summary table for the text
for tau in (0.1, 2.0):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"results/results_dw_tau{tau}_seed*.json"))]
    print(f"\nDW tau={tau}, {len(runs)} seeds; floor={np.mean([np.mean(r_['floor']) for r_ in runs]):.4f}")
    for k in STY:
        vals = np.array([[np.mean(x) for x in r_[k]] for r_ in runs])
        print(f"{k:16s}", "  ".join(f"{m:.4f}+-{s:.4f}" for m, s in zip(vals.mean(0), vals.std(0))))
    # per-x0 breakdown at K=50 for the barrier state
    j = int(np.argmin(np.abs(np.array(runs[0]["x0_test"]))))
    print("x0=0 @K=50:", {k: round(float(np.mean([r_[k][3][j] for r_ in runs])), 4) for k in STY},
          "floor", round(float(np.mean([r_["floor"][j] for r_ in runs])), 4),
          "P(x1>0|x0=0)", round(runs[0]["gt_pright"][j], 3))
