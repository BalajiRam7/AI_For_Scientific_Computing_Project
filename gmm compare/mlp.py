"""Minimal NumPy MLP (SiLU) with manual backprop and Adam -- no deep-learning framework required."""
import numpy as np


def time_feats(s, nf=4):
    s = np.asarray(s, dtype=np.float32).reshape(-1, 1)
    k = np.arange(1, nf + 1, dtype=np.float32)[None, :] * np.pi
    return np.concatenate([s, np.sin(k * s), np.cos(k * s)], 1)


class MLP:
    def __init__(self, din, dout, width=128, depth=3, seed=0):
        rng = np.random.default_rng(seed)
        sz = [din] + [width] * depth + [dout]
        self.W = [(rng.standard_normal((i, o)) / np.sqrt(i)).astype(np.float32) for i, o in zip(sz[:-1], sz[1:])]
        self.W[-1] *= 0.1
        self.b = [np.zeros(o, np.float32) for o in sz[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, X, keep=False):
        h, cache = X.astype(np.float32), []
        for W, b in zip(self.W[:-1], self.b[:-1]):
            z = h @ W + b
            sg = 1.0 / (1.0 + np.exp(-z))
            if keep:
                cache.append((h, z, sg))
            h = z * sg
        if keep:
            cache.append((h,))
            self.cache = cache
        return h @ self.W[-1] + self.b[-1]

    def step(self, X, Y, lr):
        """One Adam step on mean-squared error. Returns the loss."""
        out = self.forward(X, keep=True)
        diff = out - Y
        loss = float(np.mean(np.sum(diff ** 2, 1)))
        g = 2.0 * diff / X.shape[0]
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        h = self.cache[-1][0]
        gW[-1], gb[-1] = h.T @ g, g.sum(0)
        g = g @ self.W[-1].T
        for l in range(len(self.W) - 2, -1, -1):
            hin, z, sg = self.cache[l]
            g = g * (sg * (1 + z * (1 - sg)))
            gW[l], gb[l] = hin.T @ g, g.sum(0)
            if l > 0:
                g = g @ self.W[l].T
        self.t += 1
        b1, b2 = 0.9, 0.999
        for i, (p, gp) in enumerate(zip(self.W + self.b, gW + gb)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * gp
            self.v[i] = b2 * self.v[i] + (1 - b2) * gp * gp
            mh, vh = self.m[i] / (1 - b1 ** self.t), self.v[i] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return loss
