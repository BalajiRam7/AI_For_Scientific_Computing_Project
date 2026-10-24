"""Post-hoc analysis for Experiment 3.
(1) Expected Monte Carlo floor of each metric: mean over 5 independent exact samples of size N_SAMP
    (the single-draw floor stored in the run files is noisy because mode weights fluctuate).
(2) Spurious inter-mode mass at K = 100: fraction of forecasts whose squared Mahalanobis distance to the
    nearest mode exceeds the chi^2_2 99.9% quantile (13.8). Exact forecasts almost never land there;
    samples bridging two modes are a known failure mode of score-based models (mode interpolation)."""
import glob, json, numpy as np
import exp_gmm_jump as E

TAUS, NDRAW = [0.5, 1.0, 2.0], 5


def off_mode_fraction(x):
    d = x[:, None, :] - E.MEANS
    m2 = np.einsum("nki,kij,nkj->nk", d, E.PREC, d).min(1)
    return float((m2 > 13.8).mean())


out = {}
for tau in TAUS:
    gt = np.load(f"results/gt_gmm_tau{tau}.npz")["gt"]
    rng = np.random.default_rng(31337)
    fl = np.zeros((len(E.X0_TEST), NDRAW, 2))
    for i, v in enumerate(E.X0_TEST):
        for r in range(NDRAW):
            fl[i, r] = E.evaluate(E.simulate(np.repeat(v[None], E.N_SAMP, 0), tau, rng), gt[i])
    entry = {"floor_mean": fl.mean(1).tolist(), "floor_sd": fl.std(1).tolist(),
             "offmode_truth": [off_mode_fraction(g) for g in gt], "offmode": {}}
    for f in sorted(glob.glob(f"results/samples_gmm_tau{tau}_seed*.npz")):
        smp = np.load(f)
        for k in smp.files:
            entry["offmode"].setdefault(k, []).append([off_mode_fraction(s) for s in smp[k]])
    out[str(tau)] = entry
    print(f"tau={tau}: expected floor SW1 per x0 {fl.mean(1)[:,0].round(3)} (sd {fl.std(1)[:,0].round(3)}), "
          f"modeTV {fl.mean(1)[:,1].round(3)}")
    print("   off-mode fraction (mean over x0, seeds) truth %.4f" % np.mean(entry["offmode_truth"]),
          {k: round(float(np.mean(v)), 4) for k, v in entry["offmode"].items()})
json.dump(out, open("results/gmm_posthoc.json", "w"), indent=1)
