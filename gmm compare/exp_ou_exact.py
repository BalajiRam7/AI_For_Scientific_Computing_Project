"""Experiment 1: 2-D Ornstein-Uhlenbeck process with EXACT drifts/scores.
dX = -A X dt + dW,  A = 0.5 I + 2 J (damped rotation), stationary law N(0, I).
x1 | x0 ~ N(M x0, v I),  M = exp(-A tau),  v = 1 - exp(-tau).
Because the conditional law is Gaussian, the SI drift and the diffusion score are available in closed
form, so the error measured here is PURE time-discretisation error (no learning error).
Metric: Bures-Wasserstein-2 between the sample moments and the exact conditional, divided by sqrt(tr S).
"""
import numpy as np, json, sys
from samplers import si_coeffs, sample_si, sample_diff_sde, sample_ddim, vp_ab

a_, b_ = 0.5, 2.0
N_X0, N_SAMP = 32, 5000
KS = [5, 10, 20, 50, 100, 200]
EPSS = [0.25, 0.5, 1.0]
TAUS = [0.05, 0.25, 1.0]


def ou_M(tau):
    th = -b_ * tau
    return np.exp(-a_ * tau) * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def rel_w2(samples, m, v):
    """samples (n_x0, N, d), m (n_x0, d); exact cov v*I. Returns mean over x0 of W2/sqrt(d v)."""
    d = m.shape[-1]
    out = []
    for i in range(m.shape[0]):
        mu = samples[i].mean(0)
        C = np.cov(samples[i].T)
        ev = np.clip(np.linalg.eigvalsh(C), 0, None)
        w2sq = np.sum((mu - m[i]) ** 2) + ev.sum() + d * v - 2 * np.sqrt(v) * np.sqrt(ev).sum()
        out.append(np.sqrt(max(w2sq, 0)) / np.sqrt(d * v))
    return float(np.mean(out))


def run(tau, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((N_X0, 2))
    M = ou_M(tau)
    v = 1 - np.exp(-2 * a_ * tau)
    m = x0 @ M.T                                   # (n_x0, d)
    X0 = np.repeat(x0[:, None, :], N_SAMP, 1)      # (n_x0, N, d)
    Mb = np.repeat(m[:, None, :], N_SAMP, 1)

    res = {"tau": tau, "v": v, "floor": rel_w2(Mb + np.sqrt(v) * rng.standard_normal(X0.shape), m, v)}

    # exact SI drift for Gaussian conditional N(m, vI)
    def si_drift(eps):
        def f(x, x0_, s):
            c = si_coeffs(max(s, 1e-12), eps)
            kap = (c["db"] * c["b"] * v + c["dsg"] * c["sg"] * s) / (c["b"] ** 2 * v + c["sg"] ** 2 * max(s, 1e-12))
            return c["da"] * x0_ + c["db"] * Mb + kap * (x - c["a"] * x0_ - c["b"] * Mb)
        return f

    for eps in EPSS:
        for fol in (False, True):
            name = f"SI-{'Follmer' if fol else 'SDE'}(eps={eps})"
            res[name] = [rel_w2(sample_si(si_drift(eps), X0, K, eps, rng, follmer=fol), m, v) for K in KS]

    # exact epsilon-prediction for the VP diffusion, full state and normalised residual
    sc = np.sqrt((1 - 2 * np.exp(-a_ * tau) * np.cos(b_ * tau) + np.exp(-2 * a_ * tau)) + v)
    for resid in (False, True):
        mt = (Mb - X0) / sc if resid else Mb
        vt = v / sc ** 2 if resid else v

        def eps_fn(y, x0_, t):
            at, bt = vp_ab(t)
            return bt * (y - at * mt) / (at ** 2 * vt + bt ** 2)

        for nm, smp in (("SDE", sample_diff_sde), ("DDIM", sample_ddim)):
            name = f"Diff-{nm}{'-resid' if resid else ''}"
            out = []
            for K in KS:
                y = smp(eps_fn, X0, X0.shape, K, rng)
                out.append(rel_w2(X0 + sc * y if resid else y, m, v))
            res[name] = out
    return res


if __name__ == "__main__":
    allres = {"Ks": KS, "runs": [run(t) for t in TAUS]}
    json.dump(allres, open("results/results_ou_exact.json", "w"), indent=1)
    for r in allres["runs"]:
        print(f"\n tau={r['tau']}  cond.std={np.sqrt(r['v']):.3f}  MC floor={r['floor']:.4f}")
        for k_, val in r.items():
            if isinstance(val, list):
                print(f"{k_:24s}", " ".join(f"{x:7.4f}" for x in val))
