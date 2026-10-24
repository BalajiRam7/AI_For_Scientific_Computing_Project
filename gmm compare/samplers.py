"""Samplers shared by all experiments.

Stochastic-interpolant (SI) forecaster of Chen et al. (ICML 2024):
    I_s = alpha_s x0 + beta_s x1 + sigma_s W_s,  alpha_s = 1-s, beta_s = s^2, sigma_s = eps (1-s).
    dX_s = b_s(X_s, x0) ds + sigma_s dW_s,   X_0 = x0                 ("SI-SDE", g = sigma)
    dX_s = b^F_s(X_s, x0) ds + g^F_s dW_s,   X_0 = x0                 ("SI-Follmer", Eq. 13 of the paper)
For this schedule  g^F_s = eps*sqrt((1-s)(3-s))  and
    b^F_s = b_s (3-s)/(2-s) - [2(x-x0) + s x0] / (s(2-s)),
obtained from Eqs. (8), (10), (11), (13) of the paper.

Conditional diffusion baseline: VP-SDE (Song et al., 2021) with linear beta(t) in [0.1, 20],
epsilon-prediction, sampled with Euler-Maruyama on the reverse SDE ("Diff-SDE") or DDIM ("Diff-DDIM").
Every sampler below uses exactly K network evaluations (NFE = K).
"""
import numpy as np

BMIN, BMAX, TMIN = 0.1, 20.0, 1e-3


# ---------------- stochastic interpolant ----------------
def si_coeffs(s, eps):
    return dict(a=1 - s, da=-1.0, b=s ** 2, db=2 * s, sg=eps * (1 - s), dsg=-eps)


def follmer_drift(b, x, x0, s):
    """Post-hoc Follmer drift b^F from a drift estimate b (valid for alpha=1-s, beta=s^2, sigma=eps(1-s))."""
    if s == 0.0:  # X_0 = x0 exactly, so the 2(x-x0)/s term vanishes
        return b * 1.5 - x0 / 2.0
    return b * (3 - s) / (2 - s) - (2 * (x - x0) + s * x0) / (s * (2 - s))


def sample_si(drift_fn, x0, K, eps, rng, follmer=False):
    """drift_fn(x, x0, s) -> b_s(x, x0).  x0: array (..., d). Returns X_1 with the same shape."""
    h = 1.0 / K
    x = x0.copy()
    for k in range(K):
        s = k * h
        b = drift_fn(x, x0, s)
        if follmer:
            b = follmer_drift(b, x, x0, s)
            g = eps * np.sqrt((1 - s) * (3 - s))
        else:
            g = eps * (1 - s)
        x = x + b * h + g * np.sqrt(h) * rng.standard_normal(x.shape)
    return x


# ---------------- VP diffusion ----------------
def vp_beta(t):
    return BMIN + t * (BMAX - BMIN)


def vp_ab(t):
    """y_t = a_t y_0 + b_t eps."""
    a = np.exp(-0.25 * t ** 2 * (BMAX - BMIN) - 0.5 * t * BMIN)
    return a, np.sqrt(1 - a ** 2)


def sample_diff_sde(eps_fn, x0, d_shape, K, rng):
    """Reverse-time VP SDE, Euler-Maruyama on a uniform grid t: 1 -> TMIN. eps_fn(y, x0, t)."""
    ts = np.linspace(1.0, TMIN, K + 1)
    y = rng.standard_normal(d_shape)
    for k in range(K):
        t, h = ts[k], ts[k] - ts[k + 1]
        _, bt = vp_ab(t)
        score = -eps_fn(y, x0, t) / bt
        bet = vp_beta(t)
        y = y + (0.5 * bet * y + bet * score) * h + np.sqrt(bet * h) * rng.standard_normal(y.shape)
    return y


def sample_ddim(eps_fn, x0, d_shape, K, rng):
    """Deterministic DDIM (eta=0) on a uniform grid t: 1 -> 0; last step returns the x0-prediction."""
    ts = np.linspace(1.0, 0.0, K + 1)
    y = rng.standard_normal(d_shape)
    for k in range(K):
        a, b = vp_ab(ts[k])
        e = eps_fn(y, x0, ts[k])
        xhat = (y - b * e) / a
        a2, b2 = vp_ab(ts[k + 1]) if k < K - 1 else (1.0, 0.0)
        y = a2 * xhat + b2 * e
    return y
