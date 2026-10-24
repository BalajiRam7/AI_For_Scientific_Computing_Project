"""Experiment 3: the multi-modal jump-diffusion ("gmm") example of Chen et al. (ICML 2024),
Sec. 4.1 and App. C.1, used to test conventional CONDITIONAL DIFFUSION models against the
stochastic-interpolant (SI) forecaster on the same data, network, training budget and NFE.

System (App. C.1). rho_GMM is an equal-weight mixture of 5 Gaussians: N(m0, C0) with m0 = [5, 0],
C0 = diag(1.5, 0.1), and its rotations by 2*pi*k/5. Between jumps the particle follows Langevin
dynamics  dx = grad log rho_GMM(x) dt + sqrt(2) dW ; at Poisson times (rate lambda = 2) it is rotated
counter-clockwise by 2*pi/5. Time step dt = 0.01. rho_GMM is invariant, and the forecast law
rho_c(x_tau | x0) is multi-modal: it leaks counter-clockwise into neighbouring modes as tau grows.
(The paper's exact update order is not given in the snippet we could access; we take an
Euler-Maruyama step followed by a rotation with probability 1 - exp(-lambda dt).)

Methods (identical MLP 3x128, Adam, STEPS steps, batch 512; all samplers use exactly K NFE):
  SI-SDE, SI-Follmer         stochastic-interpolant forecaster started at x0 (samplers.py)
  Diff-SDE, Diff-DDIM        VP conditional diffusion on the full state x1, x0 as network input
  Diff-SDE-resid, -DDIM-resid  same, but diffusing the normalised increment (x1 - x0)/c
All models work in coordinates rescaled by the stationary standard deviation.

Metrics, per test x0 (ground truth = 2e4 simulated paths of the same discretised process):
  SW1      sliced 1-Wasserstein distance in the plane (64 fixed directions)
  mode-TV  total variation between predicted and true mode-occupation probabilities
           (mode = nearest of the 5 angular sectors), i.e. how well jump probabilities are captured
Usage: python exp_gmm_jump.py TAU [SEED]      (TAU in {0.5, 1.0, 2.0} as in the paper's slides)
"""
import numpy as np, json, sys, time
from scipy.stats import wasserstein_distance
from mlp import MLP, time_feats
from samplers import si_coeffs, sample_si, sample_diff_sde, sample_ddim, vp_ab, TMIN

# ------------------------------------------------------------------ system (App. C.1)
LAM, DT, EPS_SI = 2.0, 0.01, 1.0
N_TRAIN, STEPS, BATCH, LR = 100_000, 15_000, 512, 2e-3
N_GT, N_SAMP = 20_000, 2_000
KS = [10, 20, 50, 100, 200]


def rot(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


R5 = rot(2 * np.pi / 5)
M0, C0 = np.array([5.0, 0.0]), np.diag([1.5, 0.1])
MEANS = np.stack([np.linalg.matrix_power(R5, k) @ M0 for k in range(5)])
COVS = np.stack([np.linalg.matrix_power(R5, k) @ C0 @ np.linalg.matrix_power(R5, k).T for k in range(5)])
PREC = np.linalg.inv(COVS)
CHOL = np.linalg.cholesky(COVS)
LOGDET = np.log(np.linalg.det(COVS))

# test initial conditions: mode centre, between two modes, near the origin, outer tail of a mode
X0_TEST = np.array([[5.0, 0.0], [5 * np.cos(np.pi / 5), 5 * np.sin(np.pi / 5)], [1.0, 0.0], [8.0, 0.0]])
X0_NAMES = ["mode centre (5,0)", "between modes", "near origin (1,0)", "outer tail (8,0)"]


def gmm_score(x):
    """grad log rho_GMM for x of shape (..., 2)."""
    d = x[..., None, :] - MEANS                                   # (..., 5, 2)
    Pd = np.einsum("kij,...kj->...ki", PREC, d)
    logw = -0.5 * np.sum(d * Pd, -1) - 0.5 * LOGDET
    w = np.exp(logw - logw.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return -np.einsum("...k,...ki->...i", w, Pd)


def sample_gmm(n, rng):
    k = rng.integers(0, 5, n)
    return MEANS[k] + np.einsum("nij,nj->ni", CHOL[k], rng.standard_normal((n, 2)))


def simulate(x, tau, rng):
    x = np.array(x, dtype=np.float64)
    pj = 1 - np.exp(-LAM * DT)
    for _ in range(int(round(tau / DT))):
        x = x + gmm_score(x) * DT + np.sqrt(2 * DT) * rng.standard_normal(x.shape)
        j = rng.random(x.shape[0]) < pj
        x[j] = x[j] @ R5.T
    return x


# ------------------------------------------------------------------ metrics
DIRS = np.random.default_rng(2024).standard_normal((64, 2))
DIRS /= np.linalg.norm(DIRS, axis=1, keepdims=True)


def sliced_w1(a, b):
    return float(np.mean([wasserstein_distance(a @ u, b @ u) for u in DIRS]))


def mode_probs(x):
    k = np.round(np.arctan2(x[:, 1], x[:, 0]) / (2 * np.pi / 5)).astype(int) % 5
    return np.bincount(k, minlength=5) / len(x)


def evaluate(samples, gt):
    """samples, gt: (N, 2) arrays for ONE test x0. Returns (SW1, mode-TV). Usable on external samples,
    e.g. forecasts saved from the authors' gmm notebook, to compare on exactly the same footing."""
    return sliced_w1(samples, gt), float(0.5 * np.abs(mode_probs(samples) - mode_probs(gt)).sum())


# ------------------------------------------------------------------ training
def lr_at(i):
    return 1e-4 + 0.5 * (LR - 1e-4) * (1 + np.cos(np.pi * i / STEPS))


def train(kind, u0, u1, seed, rscale=1.0):
    rng = np.random.default_rng(seed + 1000)
    net = MLP(4 + 9, 2, seed=seed)
    for i in range(STEPS):
        idx = rng.integers(0, len(u0), BATCH)
        a0, a1 = u0[idx], u1[idx]
        if kind == "si":
            s = rng.random((BATCH, 1)); z = rng.standard_normal((BATCH, 2))
            c = si_coeffs(s, EPS_SI)
            inp = c["a"] * a0 + c["b"] * a1 + c["sg"] * np.sqrt(s) * z
            tgt = c["da"] * a0 + c["db"] * a1 + c["dsg"] * np.sqrt(s) * z
            tt = s
        else:
            tt = TMIN + (1 - TMIN) * rng.random((BATCH, 1)); e = rng.standard_normal((BATCH, 2))
            at, bt = vp_ab(tt)
            y1 = a1 if kind == "diff" else (a1 - a0) / rscale
            inp, tgt = at * y1 + bt * e, e
        net.step(np.concatenate([inp, a0, time_feats(tt)], 1), tgt, lr_at(i))
    return net


def field(net):
    def f(x, x0, s):
        return net.forward(np.concatenate([x, x0, time_feats(np.full(len(x), s))], 1)).astype(np.float64)
    return f


# ------------------------------------------------------------------ main
def main(tau, seed=0):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    x0 = sample_gmm(N_TRAIN, rng); x1 = simulate(x0, tau, rng)
    SC = float(x0.std())                                   # global rescaling of the state
    u0, u1 = x0 / SC, x1 / SC
    rscale = float((u1 - u0).std())
    gt = [simulate(np.repeat(v[None], N_GT, 0), tau, np.random.default_rng(500 + i)) for i, v in enumerate(X0_TEST)]
    np.savez_compressed(f"results/gt_gmm_tau{tau}.npz", x0_test=X0_TEST, gt=np.stack(gt))
    U0 = np.repeat(X0_TEST / SC, N_SAMP, 0)               # (n_x0 * N_SAMP, 2)

    def score(xs):
        xs = xs.reshape(len(X0_TEST), N_SAMP, 2)
        return [evaluate(xs[i], gt[i]) for i in range(len(X0_TEST))]

    res = {"tau": tau, "seed": seed, "Ks": KS, "x0_test": X0_TEST.tolist(), "x0_names": X0_NAMES,
           "scale": SC, "gt_mode_probs": [mode_probs(g).tolist() for g in gt]}
    res["floor"] = score(np.concatenate([simulate(np.repeat(v[None], N_SAMP, 0), tau, rng) for v in X0_TEST]))
    srng = np.random.default_rng(seed + 7)
    keep = {}   # samples at K = 100 for plotting
    si = field(train("si", u0, u1, seed))
    for nm, fol in (("SI-SDE", False), ("SI-Follmer", True)):
        out = []
        for K in KS:
            xs = SC * sample_si(si, U0, K, EPS_SI, srng, follmer=fol)
            out.append(score(xs))
            if K == 100: keep[nm] = xs
        res[nm] = out
    for kind, tag in (("diff", ""), ("resid", "-resid")):
        ef = field(train(kind, u0, u1, seed, rscale))
        for nm, smp in (("Diff-SDE", sample_diff_sde), ("Diff-DDIM", sample_ddim)):
            out = []
            for K in KS:
                y = smp(ef, U0, U0.shape, K, srng)
                xs = SC * (y if kind == "diff" else U0 + rscale * y)
                out.append(score(xs))
                if K == 100: keep[nm + tag] = xs
            res[nm + tag] = out
    np.savez_compressed(f"results/samples_gmm_tau{tau}_seed{seed}.npz", **{k: v.reshape(len(X0_TEST), N_SAMP, 2) for k, v in keep.items()})
    res["time_s"] = time.time() - t0
    json.dump(res, open(f"results/results_gmm_tau{tau}_seed{seed}.json", "w"))
    print(f"tau={tau} seed={seed} done in {res['time_s']:.0f}s   (columns: K = {KS})")
    print("floor   SW1 %.3f  modeTV %.3f" % tuple(np.mean(res["floor"], 0)))
    for k, v in res.items():
        if k.startswith(("SI", "Diff")):
            m = np.array(v).mean(1)        # (K, 2) averaged over x0
            print(f"{k:16s} SW1 " + " ".join(f"{a:.3f}" for a in m[:, 0]) + " | modeTV " + " ".join(f"{a:.3f}" for a in m[:, 1]))


if __name__ == "__main__":
    main(float(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0)
