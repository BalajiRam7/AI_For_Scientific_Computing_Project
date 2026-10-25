import numpy as np
from typing import Callable


class AdvectionDiffusionEquation2D:
    """u_t + ux u_x + uy u_y = nu (u_xx + u_yy)"""

    def __init__(
        self,
        Lx: float = 1.0,
        Ly: float = 1.0,
        T: float = 1.0,
        n_x: int = 128,
        n_y: int = 128,
        n_t: int = 100,
        velocity: dict | None = None,
        diffusivity: dict | None = None,
        ic: dict | None = None,
        source: dict | None = None,
        CFL: float = 0.5,
        second_order: bool = True,
        bc: dict | None = None,
        advection_scheme: str | None = None,
        fixed_dt_hz: float = 1000.0,
    ) -> None:
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.T = float(T)
        self.Nx = int(n_x)
        self.Ny = int(n_y)
        self.n_t = int(n_t)
        self.CFL = float(CFL)
        self.second_order = bool(second_order)

        vel_cfg = velocity or {}
        if ("v_x" in vel_cfg) or ("v_y" in vel_cfg):
            self.ux = float(vel_cfg.get("v_x", 0.0))
            self.uy = float(vel_cfg.get("v_y", 0.0))
        else:
            vel_val = vel_cfg.get("value", [0.0, 0.0])
            if isinstance(vel_val, (list, tuple)):
                self.ux = float(vel_val[0])
                self.uy = float(vel_val[1] if len(vel_val) > 1 else 0.0)
            else:
                self.ux = float(vel_val)
                self.uy = 0.0

        self.nu = float((diffusivity or {}).get("value", 0.0))
        self.ic = ic
        self.source_cfg = source
        self.bc_cfg = bc
        self.advection_scheme = (advection_scheme or "").lower() or None
        self.fixed_dt_hz = float(fixed_dt_hz)

        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dx_inv = 1.0 / self.dx
        self.dy_inv = 1.0 / self.dy

        xe = np.linspace(0.0, self.Lx, self.Nx + 1)
        ye = np.linspace(0.0, self.Ly, self.Ny + 1)
        xc = xe[:-1] + 0.5 * self.dx
        yc = ye[:-1] + 0.5 * self.dy
        self.Xc, self.Yc = np.meshgrid(xc, yc, indexing="xy")

    @staticmethod
    def _vl_limiter(
        a: np.ndarray, b: np.ndarray, c: np.ndarray, alpha: float = 2.0
    ) -> np.ndarray:
        return (
            np.sign(c)
            * (0.5 + 0.5 * np.sign(a * b))
            * np.minimum(alpha * np.minimum(np.abs(a), np.abs(b)), np.abs(c))
        )

    def _limiting_axis(
        self, u_g: np.ndarray, axis: int, N_main: int, if_second: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if axis == 1:
            a = u_g[:, 1 : N_main + 3] - u_g[:, 0 : N_main + 2]
            b = u_g[:, 2 : N_main + 4] - u_g[:, 1 : N_main + 3]
            c = 0.5 * (u_g[:, 2 : N_main + 4] - u_g[:, 0 : N_main + 2])
            gradu = self._vl_limiter(a, b, c) * if_second
            uL = np.zeros_like(u_g)
            uR = np.zeros_like(u_g)
            uL[:, 1 : N_main + 3] = u_g[:, 1 : N_main + 3] - 0.5 * gradu
            uR[:, 1 : N_main + 3] = u_g[:, 1 : N_main + 3] + 0.5 * gradu
            return uL, uR
        else:
            a = u_g[1 : N_main + 3, :] - u_g[0 : N_main + 2, :]
            b = u_g[2 : N_main + 4, :] - u_g[1 : N_main + 3, :]
            c = 0.5 * (u_g[2 : N_main + 4, :] - u_g[0 : N_main + 2, :])
            gradu = self._vl_limiter(a, b, c) * if_second
            uL = np.zeros_like(u_g)
            uR = np.zeros_like(u_g)
            uL[1 : N_main + 3, :] = u_g[1 : N_main + 3, :] - 0.5 * gradu
            uR[1 : N_main + 3, :] = u_g[1 : N_main + 3, :] + 0.5 * gradu
            return uL, uR

    def _bc2d_periodic(self, u: np.ndarray) -> np.ndarray:
        Ny, Nx = u.shape
        u_g = np.zeros((Ny + 4, Nx + 4), dtype=u.dtype)
        u_g[2 : Ny + 2, 2 : Nx + 2] = u
        u_g[2 : Ny + 2, 0:2] = u[:, -2:]
        u_g[2 : Ny + 2, Nx + 2 : Nx + 4] = u[:, 0:2]
        u_g[0:2, :] = u_g[Ny : Ny + 2, :]
        u_g[Ny + 2 : Ny + 4, :] = u_g[2:4, :]
        return u_g

    def _parse_bc_kind_and_values(self) -> tuple[str, dict[str, float | None]]:
        if not self.bc_cfg:
            return "periodic", {s: None for s in ("left", "right", "bottom", "top")}
        bc_type = str(self.bc_cfg.get("type", "periodic")).lower()
        if bc_type == "dirichletconstant":
            locs = self.bc_cfg.get("location", "left,right,top,bottom")
            if isinstance(locs, str):
                locs = [s.strip().lower() for s in locs.split(",") if s.strip()]
            val = float(self.bc_cfg.get("value", 0.0))
            values = {
                s: (val if s in locs else None)
                for s in ("left", "right", "bottom", "top")
            }
            return "dirichletconstant", values
        return "periodic", {s: None for s in ("left", "right", "bottom", "top")}

    def _bc2d_dirichlet_constant(
        self, u: np.ndarray, values: dict[str, float | None]
    ) -> np.ndarray:
        Ny, Nx = u.shape
        u_g = np.zeros((Ny + 4, Nx + 4), dtype=u.dtype)
        u_g[2 : Ny + 2, 2 : Nx + 2] = u

        left = values.get("left")
        right = values.get("right")
        bottom = values.get("bottom")
        top = values.get("top")

        if left is None:
            u_g[2 : Ny + 2, 0:2] = u[:, -2:]
        if right is None:
            u_g[2 : Ny + 2, Nx + 2 : Nx + 4] = u[:, 0:2]

        if bottom is None:
            u_g[Ny + 2 : Ny + 4, 2 : Nx + 2] = u[0:2, :]
        if top is None:
            u_g[0:2, 2 : Nx + 2] = u[-2:, :]

        if left is not None:
            u_g[2 : Ny + 2, 1] = 2.0 * float(left) - u[:, 0]
            u_g[2 : Ny + 2, 0] = 2.0 * float(left) - u[:, 1 if Nx > 1 else 0]
        if right is not None:
            u_g[2 : Ny + 2, Nx + 2] = 2.0 * float(right) - u[:, Nx - 1]
            u_g[2 : Ny + 2, Nx + 3] = (
                2.0 * float(right) - u[:, Nx - 2 if Nx > 1 else Nx - 1]
            )

        if bottom is not None:
            u_g[Ny + 2, 2 : Nx + 2] = 2.0 * float(bottom) - u[0, :]
            u_g[Ny + 3, 2 : Nx + 2] = 2.0 * float(bottom) - u[1 if Ny > 1 else 0, :]
        if top is not None:
            u_g[1, 2 : Nx + 2] = 2.0 * float(top) - u[Ny - 1, :]
            u_g[0, 2 : Nx + 2] = 2.0 * float(top) - u[Ny - 2 if Ny > 1 else Ny - 1, :]

        return u_g

    def _init_condition(self) -> np.ndarray:
        Ny, Nx = self.Ny, self.Nx
        if not self.ic:
            return np.zeros((Ny, Nx), dtype=float)
        kind = str(self.ic.get("type", "gaussian")).lower()
        if kind == "gaussian":
            A = float(self.ic.get("A", 1.0))
            sigx = float(self.ic.get("sigma_x", self.ic.get("sigma", 0.1)))
            sigy = float(self.ic.get("sigma_y", self.ic.get("sigma", 0.1)))

            if "center_x" in self.ic:
                cx = float(self.ic.get("center_x"))
            elif "rel_center_x" in self.ic or "center_rel_x" in self.ic:
                relx = float(self.ic.get("rel_center_x", self.ic.get("center_rel_x")))
                cx = (0.5 + relx) * self.Lx
            elif "deviation_x" in self.ic:
                cx = 0.5 * self.Lx + float(self.ic.get("deviation_x")) * self.Lx
            else:
                cx = 0.5 * self.Lx

            if "center_y" in self.ic:
                cy = float(self.ic.get("center_y"))
            elif "rel_center_y" in self.ic or "center_rel_y" in self.ic:
                rely = float(self.ic.get("rel_center_y", self.ic.get("center_rel_y")))
                cy = (0.5 + rely) * self.Ly
            elif "deviation_y" in self.ic:
                cy = 0.5 * self.Ly + float(self.ic.get("deviation_y")) * self.Ly
            else:
                cy = 0.5 * self.Ly

            return A * np.exp(
                -((self.Xc - cx) ** 2) / (2 * sigx**2)
                - ((self.Yc - cy) ** 2) / (2 * sigy**2)
            )
        elif kind in ("grf", "gaussian_random_field", "gp"):
            kernel = str(self.ic.get("kernel", "rbf")).lower()
            if kernel not in ("rbf", "squared_exponential", "se"):
                kernel = "rbf"

            if "ell_x" in self.ic or "ell_y" in self.ic:
                ell_x = float(self.ic.get("ell_x", self.ic.get("length_scale", 0.1)))
                ell_y = float(self.ic.get("ell_y", self.ic.get("length_scale", 0.1)))
            else:
                ell = float(self.ic.get("ell", self.ic.get("length_scale", 0.1)))
                ell_x = ell_y = ell
            ell_x = max(1.0e-6, ell_x)
            ell_y = max(1.0e-6, ell_y)

            if "variance" in self.ic:
                variance = float(self.ic.get("variance"))
                variance = max(0.0, variance)
            else:
                std = self.ic.get(
                    "std",
                    self.ic.get("sigma_f", self.ic.get("amplitude", self.ic.get("A", 1.0))),
                )
                std = max(0.0, float(std))
                variance = std * std

            mean_val = float(self.ic.get("mean", 0.0))
            seed = self.ic.get("seed", None)
            rng = np.random.default_rng(None if seed is None else int(seed))

            Ny, Nx = self.Ny, self.Nx
            if variance <= 0.0:
                return np.full((Ny, Nx), mean_val, dtype=float)

            x_coords = self.Xc[0, :]
            y_coords = self.Yc[:, 0]

            dx = x_coords[None, :] - x_coords[:, None]
            dy = y_coords[None, :] - y_coords[:, None]
            Kx = np.exp(-0.5 * (dx * dx) / (ell_x * ell_x))
            Ky = np.exp(-0.5 * (dy * dy) / (ell_y * ell_y))

            jitter_x = 1.0e-10 + 1.0e-10 * np.max(np.abs(Kx))
            jitter_y = 1.0e-10 + 1.0e-10 * np.max(np.abs(Ky))
            Kx.flat[:: Nx + 1] += jitter_x
            Ky.flat[:: Ny + 1] += jitter_y

            try:
                Lx = np.linalg.cholesky(Kx)
            except np.linalg.LinAlgError:
                Kx.flat[:: Nx + 1] += 1.0e-8
                Lx = np.linalg.cholesky(Kx)
            try:
                Ly = np.linalg.cholesky(Ky)
            except np.linalg.LinAlgError:
                Ky.flat[:: Ny + 1] += 1.0e-8
                Ly = np.linalg.cholesky(Ky)

            Z = rng.standard_normal((Ny, Nx))
            u0 = mean_val + np.sqrt(variance) * (Ly @ Z @ Lx.T)

            if bool(self.ic.get("normalize", False)):
                m = float(np.max(np.abs(u0)))
                if m > 1.0e-12:
                    u0 = u0 / m
            return u0
        elif kind == "sinusoidal":
            Ax = float(self.ic.get("Ax", 1.0))
            Ay = float(self.ic.get("Ay", 1.0))
            kx = float(self.ic.get("kx", 1.0))
            ky = float(self.ic.get("ky", 1.0))
            ph = float(self.ic.get("phase", 0.0))
            return (
                Ax
                * np.sin(2 * np.pi * kx * self.Xc / self.Lx + ph)
                * Ay
                * np.sin(2 * np.pi * ky * self.Yc / self.Ly + ph)
            )
        elif kind == "constant":
            value = float(self.ic.get("value", 0.0))
            return np.full((Ny, Nx), value, dtype=float)
        elif kind == "expression":
            expr = self.ic.get("expr", "0.0")
            return eval(
                expr,
                {
                    "np": np,
                    "x": self.Xc,
                    "y": self.Yc,
                    "Lx": self.Lx,
                    "Ly": self.Ly,
                    "T": self.T,
                    "t": 0.0,
                },
            )
        return np.zeros((Ny, Nx), dtype=float)

    def _build_source(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray] | None:
        if not self.source_cfg:
            return None
        kind = str(self.source_cfg.get("type", "constant")).lower()
        if kind == "constant":
            val = float(self.source_cfg.get("value", 0.0))
            return lambda X, Y, t: np.full_like(X, val, dtype=float)
        elif kind == "expression":
            expr = self.source_cfg.get("expr", "0.0")
            return lambda X, Y, t: eval(
                expr,
                {
                    "np": np,
                    "x": X,
                    "y": Y,
                    "t": t,
                    "Lx": self.Lx,
                    "Ly": self.Ly,
                    "T": self.T,
                },
            )
        return None

    def _flux_faces_x(self, u: np.ndarray) -> np.ndarray:
        Ny, Nx = self.Ny, self.Nx
        kind, values = self._parse_bc_kind_and_values()
        if kind == "dirichletconstant":
            u_g = self._bc2d_dirichlet_constant(u, values)
        else:
            u_g = self._bc2d_periodic(u)
        if_second = 1.0 if self.second_order else 0.0
        uLx, uRx = self._limiting_axis(u_g, axis=1, N_main=Nx, if_second=if_second)

        rows = slice(2, Ny + 2)
        cols_R = slice(1, Nx + 2)
        cols_L = slice(2, Nx + 3)

        f_adv = 0.5 * (
            self.ux * uRx[rows, cols_R]
            + self.ux * uLx[rows, cols_L]
            - abs(self.ux) * (uRx[rows, cols_R] - uLx[rows, cols_L])
        )
        if kind == "dirichletconstant" and self.ux != 0.0:
            if self.ux > 0.0 and values.get("left") is not None:
                f_adv[:, 0] = self.ux * float(values["left"])
            if self.ux < 0.0 and values.get("right") is not None:
                f_adv[:, Nx] = self.ux * float(values["right"])

        f_diff = (
            -self.nu * (u_g[rows, 2 : Nx + 3] - u_g[rows, 1 : Nx + 2]) * self.dx_inv
        )

        return f_adv + f_diff

    def _flux_faces_y(self, u: np.ndarray) -> np.ndarray:
        Ny, Nx = self.Ny, self.Nx
        kind, values = self._parse_bc_kind_and_values()
        if kind == "dirichletconstant":
            u_g = self._bc2d_dirichlet_constant(u, values)
        else:
            u_g = self._bc2d_periodic(u)
        if_second = 1.0 if self.second_order else 0.0
        uLy, uRy = self._limiting_axis(u_g, axis=0, N_main=Ny, if_second=if_second)

        cols = slice(2, Nx + 2)
        rows_R = slice(1, Ny + 2)
        rows_L = slice(2, Ny + 3)

        f_adv = 0.5 * (
            self.uy * uRy[rows_R, cols]
            + self.uy * uLy[rows_L, cols]
            - abs(self.uy) * (uRy[rows_R, cols] - uLy[rows_L, cols])
        )
        if kind == "dirichletconstant" and self.uy != 0.0:
            if self.uy > 0.0 and values.get("bottom") is not None:
                f_adv[0, :] = self.uy * float(values["bottom"])
            if self.uy < 0.0 and values.get("top") is not None:
                f_adv[Ny, :] = self.uy * float(values["top"])

        f_diff = (
            -self.nu * (u_g[2 : Ny + 3, cols] - u_g[1 : Ny + 2, cols]) * self.dy_inv
        )

        return f_adv + f_diff

    def _update(
        self, u: np.ndarray, u_tmp: np.ndarray, dt: float, t_now: float, source_fn
    ) -> np.ndarray:
        fx = self._flux_faces_x(u_tmp)
        fy = self._flux_faces_y(u_tmp)
        dudt = -self.dx_inv * (fx[:, 1:] - fx[:, :-1]) - self.dy_inv * (
            fy[1:, :] - fy[:-1, :]
        )
        if source_fn is not None:
            dudt = dudt + source_fn(self.Xc, self.Yc, t_now)
        return u + dt * dudt

    def _dt_adv(self) -> float:
        denom = 0.0
        if abs(self.ux) > 0.0:
            denom += abs(self.ux) / (self.dx + 1e-12)
        if abs(self.uy) > 0.0:
            denom += abs(self.uy) / (self.dy + 1e-12)
        if denom <= 0.0:
            return np.inf
        return self.CFL / (denom + 1e-12)

    def _dt_diff(self) -> float:
        if self.nu <= 0.0:
            return np.inf
        return self.CFL * min(
            0.5 * self.dx * self.dx / (self.nu + 1e-8),
            0.5 * self.dy * self.dy / (self.nu + 1e-8),
        )

    def _update_advection_fdm2_euler(self, u: np.ndarray, dt: float) -> np.ndarray:
        if self.bc_cfg and str(self.bc_cfg.get("type", "periodic")).lower() != "periodic":
            return self._update(u, u, dt, 0.0, None)

        ux, uy = self.ux, self.uy
        dx, dy = self.dx, self.dy

        if ux >= 0.0:
            du_dx = (3 * u - 4 * np.roll(u, 1, axis=1) + np.roll(u, 2, axis=1)) / (
                2.0 * dx
            )
        else:
            du_dx = (-3 * u + 4 * np.roll(u, -1, axis=1) - np.roll(u, -2, axis=1)) / (
                2.0 * dx
            )

        if uy >= 0.0:
            du_dy = (3 * u - 4 * np.roll(u, 1, axis=0) + np.roll(u, 2, axis=0)) / (
                2.0 * dy
            )
        else:
            du_dy = (-3 * u + 4 * np.roll(u, -1, axis=0) - np.roll(u, -2, axis=0)) / (
                2.0 * dy
            )

        return u - dt * (ux * du_dx + uy * du_dy)

    def solve(self) -> np.ndarray:
        dt_save = self.T / max(1, self.n_t)
        it_tot = int(np.ceil(self.T / dt_save)) + 1

        u = self._init_condition()
        source_fn = self._build_source()

        Ny, Nx = self.Ny, self.Nx
        uu = np.zeros((it_tot, Ny, Nx), dtype=float)

        t = 0.0
        tsave = 0.0
        i_save = 0
        uu[i_save] = u

        if abs(self.ux) == 0.0 and abs(self.uy) == 0.0 and self.nu <= 0.0:
            while i_save < it_tot - 1:
                i_save += 1
                uu[i_save] = u
            return uu

        use_fdm2 = (
            (self.advection_scheme == "fdm2_euler_fixed") and self.nu <= 0.0
        )

        while t < self.T - 1e-15:
            while t >= tsave - 1e-15 and i_save < it_tot - 1:
                if i_save > 0:
                    uu[i_save] = u
                i_save += 1
                tsave += dt_save

            if use_fdm2:
                dt_fixed = 1.0 / max(1.0, self.fixed_dt_hz)
                dt_cfl = self._dt_adv()
                dt = min(dt_fixed, dt_cfl, self.T - t, tsave - t)
            else:
                dt = min(self._dt_adv(), self._dt_diff(), self.T - t, tsave - t)
                if not np.isfinite(dt) or dt <= 1e-12:
                    dt = min(self.T - t, max(0.0, tsave - t))
            if dt <= 1e-12:
                break

            if use_fdm2 and source_fn is None:
                u = self._update_advection_fdm2_euler(u, dt)
            else:
                u_tmp = self._update(u, u, 0.5 * dt, t + 0.5 * dt, source_fn)
                u = self._update(u, u_tmp, dt, t + 0.5 * dt, source_fn)

            t += dt

        uu[-1] = u
        return uu


if __name__ == "__main__":
    eq = AdvectionDiffusionEquation2D(
        Lx=1.0,
        Ly=1.0,
        T=1.0,
        n_x=128,
        n_y=128,
        n_t=99,
        velocity={"v_x": 0.1, "v_y": -0.2},
        diffusivity={"type": "constant", "value": 0.1},
        ic={
            "type": "gaussian",
            "A": 1.0,
            "sigma": 0.1,
            "sigma_x": 0.1,
            "sigma_y": 0.1,
            "center_x": 0.5,
            "center_y": 0.5,
        },
        CFL=0.5,
        second_order=True,
        bc={"type": "Periodic"},
    )

    snapshots = eq.solve()
    print(snapshots.shape)
