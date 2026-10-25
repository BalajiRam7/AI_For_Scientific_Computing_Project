import numpy as np
from typing import Callable


class BurgersEquation1D:
    """u_t + (u^2/2)_x = nu u_xx"""

    def __init__(
        self,
        L: float = 1.0,
        T: float = 1.0,
        n_x: int = 256,
        n_t: int = 100,
        viscosity: dict | None = None,
        ic: dict | None = None,
        source: dict | None = None,
        CFL: float = 0.5,
        second_order: bool = True,
        bc: dict | None = None,
    ) -> None:
        self.L = float(L)
        self.T = float(T)
        self.N = int(n_x)
        self.n_t = int(n_t)
        self.CFL = float(CFL)
        self.second_order = bool(second_order)

        cfg = viscosity
        self.nu = float((cfg or {}).get("value", 0.0))

        self.ic = ic
        self.source_cfg = source
        self.bc_cfg = bc

        self.dx = self.L / self.N
        self.dx_inv = 1.0 / self.dx

        xe = np.linspace(0.0, self.L, self.N + 1)
        xc = xe[:-1] + 0.5 * self.dx
        self.xc = xc

    @staticmethod
    def _vl_limiter(
        a: np.ndarray, b: np.ndarray, c: np.ndarray, alpha: float = 2.0
    ) -> np.ndarray:
        return (
            np.sign(c)
            * (0.5 + 0.5 * np.sign(a * b))
            * np.minimum(alpha * np.minimum(np.abs(a), np.abs(b)), np.abs(c))
        )

    def _limiting(
        self, u_g: np.ndarray, N: int, if_second: float
    ) -> tuple[np.ndarray, np.ndarray]:
        du_L = u_g[1 : N + 3] - u_g[0 : N + 2]
        du_R = u_g[2 : N + 4] - u_g[1 : N + 3]
        du_M = 0.5 * (u_g[2 : N + 4] - u_g[0 : N + 2])
        gradu = self._vl_limiter(du_L, du_R, du_M) * if_second

        uL = np.zeros_like(u_g)
        uR = np.zeros_like(u_g)
        uL[1 : N + 3] = u_g[1 : N + 3] - 0.5 * gradu
        uR[1 : N + 3] = u_g[1 : N + 3] + 0.5 * gradu
        return uL, uR

    @staticmethod
    def _bc_periodic(u: np.ndarray, N: int) -> np.ndarray:
        u_g = np.zeros(N + 4, dtype=u.dtype)
        u_g[2 : N + 2] = u
        u_g[0:2] = u[-2:]
        u_g[N + 2 : N + 4] = u[0:2]
        return u_g

    def _bc_dirichlet_constant(
        self, u: np.ndarray, N: int, left_val: float | None, right_val: float | None
    ) -> np.ndarray:
        u_g = np.zeros(N + 4, dtype=u.dtype)
        u_g[2 : N + 2] = u
        if left_val is not None:
            if N >= 2:
                u_g[1] = 2.0 * float(left_val) - u[0]
                u_g[0] = 2.0 * float(left_val) - u[1]
            else:
                u_g[1] = 2.0 * float(left_val) - u[0]
                u_g[0] = 2.0 * float(left_val) - u[0]
        else:
            u_g[0:2] = u[-2:]
        if right_val is not None:
            if N >= 2:
                u_g[N + 2] = 2.0 * float(right_val) - u[N - 1]
                u_g[N + 3] = 2.0 * float(right_val) - u[N - 2]
            else:
                u_g[N + 2] = 2.0 * float(right_val) - u[N - 1]
                u_g[N + 3] = 2.0 * float(right_val) - u[N - 1]
        else:
            u_g[N + 2 : N + 4] = u[0:2]
        return u_g

    def _get_bc_kind_and_values(self) -> tuple[str, float | None, float | None]:
        if not self.bc_cfg:
            return "periodic", None, None
        bc_type = str(self.bc_cfg.get("type", "periodic")).lower()
        if bc_type == "dirichletconstant":
            loc = self.bc_cfg.get("location", "left,right")
            if isinstance(loc, str):
                locs = [s.strip().lower() for s in loc.split(",") if s.strip()]
            else:
                locs = ["left", "right"]
            val = float(self.bc_cfg.get("value", 0.0))
            left_val = val if "left" in locs else None
            right_val = val if "right" in locs else None
            return "dirichletconstant", left_val, right_val
        return "periodic", None, None

    def _init_condition(self) -> np.ndarray:
        if not self.ic:
            return np.zeros(self.N, dtype=float)

        kind = str(self.ic.get("type", "gaussian")).lower()
        if kind == "gaussian":
            A = float(self.ic.get("A", 1.0))
            sigma = float(self.ic.get("sigma", 0.1))
            if "center" in self.ic:
                c0 = float(self.ic.get("center"))
            elif "center_rel" in self.ic:
                c0 = (0.5 + float(self.ic.get("center_rel"))) * self.L
            elif "deviation" in self.ic:
                c0 = 0.5 * self.L + float(self.ic.get("deviation")) * self.L
            else:
                c0 = 0.5 * self.L
            return A * np.exp(-((self.xc - c0) ** 2) / (2.0 * sigma**2))
        elif kind in ("grf", "gaussian_random_field", "gp"):
            kernel = str(self.ic.get("kernel", "rbf")).lower()
            if kernel not in ("rbf", "squared_exponential", "se"):
                kernel = "rbf"

            ell = self.ic.get("ell", self.ic.get("length_scale", 0.1))
            ell = max(1.0e-6, float(ell))

            if "variance" in self.ic:
                variance = float(self.ic.get("variance"))
                variance = max(0.0, variance)
            else:
                std = self.ic.get(
                    "std",
                    self.ic.get(
                        "sigma_f", self.ic.get("amplitude", self.ic.get("A", 1.0))
                    ),
                )
                std = max(0.0, float(std))
                variance = std * std

            mean_val = float(self.ic.get("mean", 0.0))
            seed = self.ic.get("seed", None)
            rng = np.random.default_rng(None if seed is None else int(seed))

            X = self.xc
            diffs = X[:, None] - X[None, :]
            K = variance * np.exp(-0.5 * (diffs * diffs) / (ell * ell))

            jitter = 1.0e-10 * max(1.0, variance) + 1.0e-12
            K.flat[:: K.shape[0] + 1] += jitter

            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K.flat[:: K.shape[0] + 1] += 1.0e-8
                L = np.linalg.cholesky(K)

            z = rng.standard_normal(self.N)
            u0 = mean_val + L @ z

            if bool(self.ic.get("normalize", False)):
                m = float(np.max(np.abs(u0)))
                if m > 1.0e-12:
                    u0 = u0 / m
            return u0
        elif kind == "sinusoidal":
            A = float(self.ic.get("A", 1.0))
            periods = float(self.ic.get("periods", 1.0))
            phase = float(self.ic.get("phase", 0.0))
            return A * np.sin(2.0 * np.pi * periods * self.xc / self.L + phase)
        elif kind == "constant":
            value = float(self.ic.get("value", 0.0))
            return np.full(self.N, value, dtype=float)
        elif kind == "expression":
            expr = self.ic.get("expr", "0.0")
            return np.array(
                [
                    eval(expr, {"np": np, "x": x, "L": self.L, "T": self.T, "t": 0.0})
                    for x in self.xc
                ],
                dtype=float,
            )
        return np.zeros(self.N, dtype=float)

    def _build_source(self) -> Callable[[np.ndarray, float], np.ndarray] | None:
        if not self.source_cfg:
            return None
        kind = str(self.source_cfg.get("type", "constant")).lower()
        if kind == "constant":
            val = float(self.source_cfg.get("value", 0.0))
            return lambda x, t: np.full_like(x, val, dtype=float)
        elif kind == "expression":
            expr = self.source_cfg.get("expr", "0.0")
            return lambda x, t: np.array(
                [
                    eval(expr, {"np": np, "x": xi, "t": t, "L": self.L, "T": self.T})
                    for xi in x
                ],
                dtype=float,
            )
        return None

    def _flux_faces(self, u_centered: np.ndarray) -> np.ndarray:
        N = self.N
        kind, left_val, right_val = self._get_bc_kind_and_values()
        if kind == "dirichletconstant":
            u_g = self._bc_dirichlet_constant(u_centered, N, left_val, right_val)
        else:
            u_g = self._bc_periodic(u_centered, N)

        if_second = 1.0 if self.second_order else 0.0
        uL, uR = self._limiting(u_g, N, if_second)

        fL = 0.5 * (uL**2)
        fR = 0.5 * (uR**2)
        f_adv = 0.5 * (
            fR[1 : N + 2]
            + fL[2 : N + 3]
            - 0.5
            * np.abs(uL[2 : N + 3] + uR[1 : N + 2])
            * (uL[2 : N + 3] - uR[1 : N + 2])
        )

        if kind == "dirichletconstant":
            if left_val is not None:
                if np.mean(u_centered[: max(1, N // 16)]) > 0.0:
                    f_adv[0] = 0.5 * float(left_val) ** 2
            if right_val is not None:
                if np.mean(u_centered[-max(1, N // 16) :]) < 0.0:
                    f_adv[N] = 0.5 * float(right_val) ** 2

        f_diff = -self.nu * (u_g[2 : N + 3] - u_g[1 : N + 2]) * self.dx_inv

        return f_adv + f_diff

    def _update(
        self, u: np.ndarray, u_tmp: np.ndarray, dt: float, t_now: float, source_fn
    ) -> np.ndarray:
        f = self._flux_faces(u_tmp)
        u_new = u - dt * self.dx_inv * (f[1 : self.N + 1] - f[0 : self.N])
        if source_fn is not None:
            u_new = u_new + dt * source_fn(self.xc, t_now)
        return u_new

    def _dt_adv(self, u: np.ndarray) -> float:
        max_speed = float(np.max(np.abs(u))) if u.size > 0 else 0.0
        return (
            (self.dx / (max_speed + 1.0e-8)) * self.CFL if max_speed > 0.0 else np.inf
        )

    def _dt_diff(self) -> float:
        return (
            (0.5 * self.dx * self.dx / (self.nu + 1.0e-8)) * self.CFL
            if self.nu > 0.0
            else np.inf
        )

    def solve(self) -> np.ndarray:
        dt_save = self.T / max(1, self.n_t)
        it_tot = int(np.ceil(self.T / dt_save)) + 1

        u = self._init_condition()
        source_fn = self._build_source()

        uu = np.zeros((it_tot, self.N), dtype=float)
        t = 0.0
        tsave = 0.0
        i_save = 0
        uu[i_save] = u

        while t < self.T - 1e-15:
            while t >= tsave - 1e-15 and i_save < it_tot - 1:
                if i_save > 0:
                    uu[i_save] = u
                i_save += 1
                tsave += dt_save

            dt = min(self._dt_adv(u), self._dt_diff(), self.T - t, tsave - t)
            if dt <= 1e-12:
                break

            u_tmp = self._update(u, u, 0.5 * dt, t + 0.5 * dt, source_fn)
            u = self._update(u, u_tmp, dt, t + 0.5 * dt, source_fn)

            t += dt

        uu[-1] = u
        return uu


if __name__ == "__main__":
    equation = BurgersEquation1D(
        L=1.0,
        T=1.0,
        n_x=200,
        n_t=99,
        viscosity={"type": "constant", "value": 0.001},
        ic={"type": "Gaussian", "A": 1.0, "sigma": 0.1, "center": 0.3},
        bc={"type": "DirichletConstant", "location": "left,right", "value": 0.0},
        CFL=0.5,
        second_order=True,
    )
    snapshots = equation.solve()

    print(snapshots.shape)
