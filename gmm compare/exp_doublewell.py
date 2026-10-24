"""Experiment 2: 1-D overdamped Langevin dynamics in a double well, LEARNED models.
dX = -V'(X) dt + sqrt(2 D) dW,  V(x) = (x^2 - 1)^2,  D = 0.3.
Pairs (x0, x1) = (X_t, X_{t+tau}) with X_t drawn from the stationary law. All three models use the same
MLP (3 x 128, SiLU), optimiser, batch size and number of steps. Ground truth conditional samples come from
fine-step (dt = 1e-3) Euler-Maruyama simulation started at each test x0.
Usage: python exp_doublewell.py TAU SEED
"""
import numpy as np, json, sys, time
from scipy.stats import wasserstein_distance
from mlp import MLP, time_feats
from samplers import si_coeffs, sample_si, sample_diff_sde, sample_ddim, vp_ab, TMIN

D = 0.3
EPS_SI = 1.0
N_TRAIN, STEPS, BATCH, LR = 50000, 4000, 512, 2e-3
X0_TEST = np.linspace(-1.2, 1.2, 9)
N_GT, N_SAMP = 20000, 2000
KS = [5, 10, 20, 50, 100]


def simulate(x0, tau, rng, dt=1e-3):
    x = np.array(x0, dtype=np.float64)
    for _ in range(int(round(tau / dt))):
        x += -4 * x * (x ** 2 - 1) * dt + np.sqrt(2 * D * dt) * rng.standard_normal(x.shape)
    return x


def stationary(n, rng):
    g = np.linspace(-2.5, 2.5, 20001)
    cdf = np.cumsum(np.exp(-(g ** 2 - 1) ** 2 / D)); cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, g)


def lr_at(i):
    return 1e-4 + 0.5 * (LR - 1e-4) * (1 + np.cos(np.pi * i / STEPS))


def train(kind, x0, x1, seed, scale=1.0):
    rng = np.random.default_rng(seed + 1000)
    net = MLP(2 + 9, 1, seed=seed)
    for i in range(STEPS):
        idx = rng.integers(0, len(x0), BATCH)
        a0, a1 = x0[idx, None], x1[idx, None]
        if kind == "si":
            s = rng.random((BATCH, 1)); z = rng.standard_normal((BATCH, 1))
            c = si_coeffs(s, EPS_SI)
            inp = c["a"] * a0 + c["b"] * a1 + c["sg"] * np.sqrt(s) * z
            tgt = c["da"] * a0 + c["db"] * a1 + c["dsg"] * np.sqrt(s) * z
            tt = s
        else:  # VP diffusion, eps-prediction; target is x1 ("full") or (x1-x0)/scale ("resid")
            tt = TMIN + (1 - TMIN) * rng.random((BATCH, 1)); e = rng.standard_normal((BATCH, 1))
            at, bt = vp_ab(tt)
            y1 = a1 if kind == "diff" else (a1 - a0) / scale
            inp, tgt = at * y1 + bt * e, e
        net.step(np.concatenate([inp, a0, time_feats(tt)], 1), tgt, lr_at(i))
    return net


def field(net):
    def f(x, x0, s):
        return net.forward(np.concatenate([x, x0, time_feats(np.full(len(x), s))], 1)).astype(np.float64)
    return f


def main(tau, seed):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    x0 = stationary(N_TRAIN, rng); x1 = simulate(x0, tau, rng)
    scale = float(np.std(x1 - x0))
    gt = [simulate(np.full(N_GT, v), tau, np.random.default_rng(123 + i)) for i, v in enumerate(X0_TEST)]
    X0 = np.repeat(X0_TEST, N_SAMP)[:, None]

    def score(samples):
        sm = samples.reshape(len(X0_TEST), N_SAMP)
        return [float(wasserstein_distance(sm[i], gt[i])) for i in range(len(X0_TEST))]

    res = {"tau": tau, "seed": seed, "Ks": KS, "x0_test": X0_TEST.tolist(),
           "gt_std": [float(g.std()) for g in gt], "gt_pright": [float((g > 0).mean()) for g in gt]}
    # Monte-Carlo floor: an independent exact sample of size N_SAMP against the ground truth
    res["floor"] = score(np.concatenate([simulate(np.full(N_SAMP, v), tau, rng) for v in X0_TEST]))
    srng = np.random.default_rng(seed + 7)
    si = field(train("si", x0, x1, seed))
    res["SI-SDE"] = [score(sample_si(si, X0, K, EPS_SI, srng)) for K in KS]
    res["SI-Follmer"] = [score(sample_si(si, X0, K, EPS_SI, srng, follmer=True)) for K in KS]
    for kind, nm in (("diff", ""), ("resid", "-resid")):
        ef = field(train(kind, x0, x1, seed, scale))
        sc = 1.0 if kind == "diff" else scale
        off = 0.0 if kind == "diff" else X0
        res["Diff-SDE" + nm] = [score(off + sc * sample_diff_sde(ef, X0, X0.shape, K, srng)) for K in KS]
        res["Diff-DDIM" + nm] = [score(off + sc * sample_ddim(ef, X0, X0.shape, K, srng)) for K in KS]
    # keep a few samples at x0 = 0 (barrier top) for plotting, K = 50
    j = int(np.argmin(np.abs(X0_TEST)))
    X0b = np.zeros((4000, 1))
    res["hist_x0"] = 0.0
    res["hist_gt"] = gt[j][:4000].tolist()
    res["hist_SI-Follmer"] = sample_si(si, X0b, 50, EPS_SI, srng, follmer=True)[:, 0].tolist()
    res["hist_Diff-DDIM"] = sample_ddim(ef, X0b, X0b.shape, 50, srng)[:, 0].tolist()  # residual model
    res["time_s"] = time.time() - t0
    json.dump(res, open(f"results/results_dw_tau{tau}_seed{seed}.json", "w"))
    print(f"tau={tau} seed={seed} done in {res['time_s']:.0f}s")
    for k, v in res.items():
        if k.startswith(("SI", "Diff")):
            print(f"{k:18s}", " ".join(f"{np.mean(x):.4f}" for x in v))
    print("floor", np.mean(res["floor"]))


if __name__ == "__main__":
    main(float(sys.argv[1]), int(sys.argv[2]))
