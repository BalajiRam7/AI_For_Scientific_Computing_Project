import numpy as np
from typing import Callable, Tuple, Optional


class BurgersEquation2D:
    """
    u_t + (0.5 u^2)_x + (u v)_y = nu (u_xx + u_yy)
    v_t + (u v)_x + (0.5 v^2)_y = nu (v_xx + v_yy)
    """

    def __init__(
        self,
        Lx: float = 1.0,
        Ly: float = 1.0,
        T: float = 1.0,
        n_x: int = 128,
        n_y: int = 128,
        n_t: int = 100,
        viscosity: Optional[dict] = None,
        ic: Optional[dict] = None,
        source: Optional[dict] = None,
        CFL: float = 0.5,
        second_order: bool = True,
        bc: Optional[dict] = None,
        advection_scheme: Optional[str] = None,
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

        self.nu = float((viscosity or {}).get("value", 0.0))
        self.ic = ic or {}
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
        self, q_g: np.ndarray, axis: int, N_main: int, if_second: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        if axis == 1:
            a = q_g[:, 1 : N_main + 3] - q_g[:, 0 : N_main + 2]
            b = q_g[:, 2 : N_main + 4] - q_g[:, 1 : N_main + 3]
            c = 0.5 * (q_g[:, 2 : N_main + 4] - q_g[:, 0 : N_main + 2])
            grad = self._vl_limiter(a, b, c) * if_second
            qL = np.zeros_like(q_g)
            qR = np.zeros_like(q_g)
            qL[:, 1 : N_main + 3] = q_g[:, 1 : N_main + 3] - 0.5 * grad
            qR[:, 1 : N_main + 3] = q_g[:, 1 : N_main + 3] + 0.5 * grad
            return qL, qR
        else:
            a = q_g[1 : N_main + 3, :] - q_g[0 : N_main + 2, :]
            b = q_g[2 : N_main + 4, :] - q_g[1 : N_main + 3, :]
            c = 0.5 * (q_g[2 : N_main + 4, :] - q_g[0 : N_main + 2, :])
            grad = self._vl_limiter(a, b, c) * if_second
            qL = np.zeros_like(q_g)
            qR = np.zeros_like(q_g)
            qL[1 : N_main + 3, :] = q_g[1 : N_main + 3, :] - 0.5 * grad
            qR[1 : N_main + 3, :] = q_g[1 : N_main + 3, :] + 0.5 * grad
            return qL, qR

    def _parse_bc_kind_and_values(self) -> Tuple[str, dict]:
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

    def _bc2d_periodic(self, q: np.ndarray) -> np.ndarray:
        Ny, Nx = q.shape
        q_g = np.zeros((Ny + 4, Nx + 4), dtype=q.dtype)
        q_g[2 : Ny + 2, 2 : Nx + 2] = q
        q_g[2 : Ny + 2, 0:2] = q[:, -2:]
        q_g[2 : Ny + 2, Nx + 2 : Nx + 4] = q[:, 0:2]
        q_g[0:2, :] = q_g[Ny : Ny + 2, :]
        q_g[Ny + 2 : Ny + 4, :] = q_g[2:4, :]
        return q_g

    def _bc2d_dirichlet_constant(self, q: np.ndarray, values: dict) -> np.ndarray:
        Ny, Nx = q.shape
        q_g = np.zeros((Ny + 4, Nx + 4), dtype=q.dtype)
        q_g[2 : Ny + 2, 2 : Nx + 2] = q
        left, right, bottom, top = [
            values.get(s) for s in ("left", "right", "bottom", "top")
        ]
        if left is None:
            q_g[2 : Ny + 2, 0:2] = q[:, -2:]
        if right is None:
            q_g[2 : Ny + 2, Nx + 2 : Nx + 4] = q[:, 0:2]
        if bottom is None:
            q_g[Ny + 2 : Ny + 4, 2 : Nx + 2] = q[0:2, :]
        if top is None:
            q_g[0:2, 2 : Nx + 2] = q[-2:, :]
        if left is not None:
            q_g[2 : Ny + 2, 1] = 2.0 * float(left) - q[:, 0]
            q_g[2 : Ny + 2, 0] = 2.0 * float(left) - q[:, 1 if Nx > 1 else 0]
        if right is not None:
            q_g[2 : Ny + 2, Nx + 2] = 2.0 * float(right) - q[:, Nx - 1]
            q_g[2 : Ny + 2, Nx + 3] = (
                2.0 * float(right) - q[:, Nx - 2 if Nx > 1 else Nx - 1]
            )
        if bottom is not None:
            q_g[Ny + 2, 2 : Nx + 2] = 2.0 * float(bottom) - q[0, :]
            q_g[Ny + 3, 2 : Nx + 2] = 2.0 * float(bottom) - q[1 if Ny > 1 else 0, :]
        if top is not None:
            q_g[1, 2 : Nx + 2] = 2.0 * float(top) - q[Ny - 1, :]
            q_g[0, 2 : Nx + 2] = 2.0 * float(top) - q[Ny - 2 if Ny > 1 else Ny - 1, :]
        return q_g

    def _init_condition(self) -> Tuple[np.ndarray, np.ndarray]:
        kind = str(self.ic.get("type", "gaussian")).lower()
        Ny, Nx = self.Ny, self.Nx

        def _gaussian(A, sigx, sigy, cx, cy):
            return A * np.exp(
                -((self.Xc - cx) ** 2) / (2 * sigx**2)
                - ((self.Yc - cy) ** 2) / (2 * sigy**2)
            )

        if kind == "gaussian":
            sigx = float(self.ic.get("sigma_x", self.ic.get("sigma", 0.1)))
            sigy = float(self.ic.get("sigma_y", self.ic.get("sigma", 0.1)))
            relx = float(self.ic.get("rel_center_x", 0.0))
            rely = float(self.ic.get("rel_center_y", 0.0))
            cx = (0.5 + relx) * self.Lx
            cy = (0.5 + rely) * self.Ly
            Au = float(self.ic.get("A_u", self.ic.get("A", 1.0)))
            Av = float(self.ic.get("A_v", 0.0))
            u0 = _gaussian(Au, sigx, sigy, cx, cy)
            v0 = _gaussian(Av, sigx, sigy, cx, cy)
            return u0, v0

        if kind == "sinusoidal":
            Ax = float(self.ic.get("Ax_u", self.ic.get("Ax", 1.0)))
            Ay = float(self.ic.get("Ay_u", self.ic.get("Ay", 1.0)))
            kx = float(self.ic.get("kx_u", self.ic.get("kx", 1.0)))
            ky = float(self.ic.get("ky_u", self.ic.get("ky", 1.0)))
            ph = float(self.ic.get("phase_u", self.ic.get("phase", 0.0)))
            u0 = (
                Ax
                * np.sin(2 * np.pi * kx * self.Xc / self.Lx + ph)
                * Ay
                * np.sin(2 * np.pi * ky * self.Yc / self.Ly + ph)
            )

            Axv = float(self.ic.get("Ax_v", 0.0))
            Ayv = float(self.ic.get("Ay_v", 0.0))
            kxv = float(self.ic.get("kx_v", 1.0))
            kyv = float(self.ic.get("ky_v", 1.0))
            phv = float(self.ic.get("phase_v", 0.0))
            v0 = (
                Axv
                * np.sin(2 * np.pi * kxv * self.Xc / self.Lx + phv)
                * Ayv
                * np.sin(2 * np.pi * kyv * self.Yc / self.Ly + phv)
            )
            return u0, v0

        if kind == "constant":
            uval = float(self.ic.get("u", 0.0))
            vval = float(self.ic.get("v", 0.0))
            return np.full((Ny, Nx), uval, float), np.full((Ny, Nx), vval, float)

        if kind == "expression":
            expr_u = self.ic.get("expr_u", "0.0")
            expr_v = self.ic.get("expr_v", "0.0")
            env = {
                "np": np,
                "x": self.Xc,
                "y": self.Yc,
                "Lx": self.Lx,
                "Ly": self.Ly,
                "T": self.T,
                "t": 0.0,
            }
            return eval(expr_u, env), eval(expr_v, env)

        if kind in ("grf", "gaussian_random_field", "gp"):
            ell_x = float(self.ic.get("ell_x", 0.1))
            ell_y = float(self.ic.get("ell_y", 0.1))
            var_u = float(self.ic.get("variance_u", 1.0))
            var_v = float(self.ic.get("variance_v", 1.0))

            x_coords = self.Xc[0, :]
            y_coords = self.Yc[:, 0]
            dx = x_coords[None, :] - x_coords[:, None]
            dy = y_coords[None, :] - y_coords[:, None]
            Kx = np.exp(-0.5 * (dx * dx) / (ell_x * ell_x)) + 1e-8 * np.eye(Nx)
            Ky = np.exp(-0.5 * (dy * dy) / (ell_y * ell_y)) + 1e-8 * np.eye(Ny)
            Lx = np.linalg.cholesky(Kx)
            Ly = np.linalg.cholesky(Ky)

            rng = np.random.default_rng(self.ic.get("seed"))
            Zu = rng.standard_normal((Ny, Nx))
            Zv = rng.standard_normal((Ny, Nx))
            u0 = np.sqrt(var_u) * (Ly @ Zu @ Lx.T)
            v0 = np.sqrt(var_v) * (Ly @ Zv @ Lx.T)
            return u0, v0

        return np.zeros((Ny, Nx), float), np.zeros((Ny, Nx), float)

    def _build_source(
        self,
    ) -> Optional[
        Callable[[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]
    ]:
        if not self.source_cfg:
            return None
        kind = str(self.source_cfg.get("type", "constant")).lower()
        if kind == "constant":
            su = float(self.source_cfg.get("u", 0.0))
            sv = float(self.source_cfg.get("v", 0.0))
            return lambda X, Y, t: (
                np.full_like(X, su, float),
                np.full_like(X, sv, float),
            )
        if kind == "expression":
            expr_u = self.source_cfg.get("expr_u", "0.0")
            expr_v = self.source_cfg.get("expr_v", "0.0")

            def _src(X, Y, t):
                env = {
                    "np": np,
                    "x": X,
                    "y": Y,
                    "t": t,
                    "Lx": self.Lx,
                    "Ly": self.Ly,
                    "T": self.T,
                }
                return eval(expr_u, env), eval(expr_v, env)

            return _src
        return None

    def _ghostify(self, q: np.ndarray) -> Tuple[str, dict, np.ndarray]:
        kind, values = self._parse_bc_kind_and_values()
        q_g = (
            self._bc2d_dirichlet_constant(q, values)
            if kind == "dirichletconstant"
            else self._bc2d_periodic(q)
        )
        return kind, values, q_g

    def _fluxes_x(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Ny, Nx = self.Ny, self.Nx
        if_second = 1.0 if self.second_order else 0.0

        _, _, ug = self._ghostify(u)
        _, _, vg = self._ghostify(v)
        uLx, uRx = self._limiting_axis(ug, axis=1, N_main=Nx, if_second=if_second)
        vLx, vRx = self._limiting_axis(vg, axis=1, N_main=Nx, if_second=if_second)

        rows = slice(2, Ny + 2)
        cols_R = slice(1, Nx + 2)
        cols_L = slice(2, Nx + 3)

        uL = uRx[rows, cols_R]
        uR = uLx[rows, cols_L]

        vL = vRx[rows, cols_R]
        vR = vLx[rows, cols_L]

        FxL_u = 0.5 * (uL**2)
        FxR_u = 0.5 * (uR**2)
        FxL_v = uL * vL
        FxR_v = uR * vR

        alpha = np.maximum(np.abs(uL), np.abs(uR))
        Fx_u = 0.5 * (FxL_u + FxR_u) - 0.5 * alpha * (uR - uL)
        Fx_v = 0.5 * (FxL_v + FxR_v) - 0.5 * alpha * (vR - vL)

        qx_u = (ug[rows, 2 : Nx + 3] - ug[rows, 1 : Nx + 2]) * self.dx_inv
        qx_v = (vg[rows, 2 : Nx + 3] - vg[rows, 1 : Nx + 2]) * self.dx_inv
        Fx_u += -self.nu * qx_u
        Fx_v += -self.nu * qx_v
        return Fx_u, Fx_v

    def _fluxes_y(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Ny, Nx = self.Ny, self.Nx
        if_second = 1.0 if self.second_order else 0.0

        _, _, ug = self._ghostify(u)
        _, _, vg = self._ghostify(v)
        uLy, uRy = self._limiting_axis(ug, axis=0, N_main=Ny, if_second=if_second)
        vLy, vRy = self._limiting_axis(vg, axis=0, N_main=Ny, if_second=if_second)

        cols = slice(2, Nx + 2)
        rows_R = slice(1, Ny + 2)
        rows_L = slice(2, Ny + 3)

        uL = uRy[rows_R, cols]
        uR = uLy[rows_L, cols]

        vL = vRy[rows_R, cols]
        vR = vLy[rows_L, cols]

        FyL_u = uL * vL
        FyR_u = uR * vR
        FyL_v = 0.5 * (vL**2)
        FyR_v = 0.5 * (vR**2)

        alpha = np.maximum(np.abs(vL), np.abs(vR))
        Fy_u = 0.5 * (FyL_u + FyR_u) - 0.5 * alpha * (uR - uL)
        Fy_v = 0.5 * (FyL_v + FyR_v) - 0.5 * alpha * (vR - vL)

        qy_u = (ug[2 : Ny + 3, cols] - ug[1 : Ny + 2, cols]) * self.dy_inv
        qy_v = (vg[2 : Ny + 3, cols] - vg[1 : Ny + 2, cols]) * self.dy_inv
        Fy_u += -self.nu * qy_u
        Fy_v += -self.nu * qy_v
        return Fy_u, Fy_v

    def _rhs(
        self, u: np.ndarray, v: np.ndarray, t_now: float, src_fn
    ) -> Tuple[np.ndarray, np.ndarray]:
        Fx_u, Fx_v = self._fluxes_x(u, v)
        Fy_u, Fy_v = self._fluxes_y(u, v)
        dudt = -self.dx_inv * (Fx_u[:, 1:] - Fx_u[:, :-1]) - self.dy_inv * (
            Fy_u[1:, :] - Fy_u[:-1, :]
        )
        dvdt = -self.dx_inv * (Fx_v[:, 1:] - Fx_v[:, :-1]) - self.dy_inv * (
            Fy_v[1:, :] - Fy_v[:-1, :]
        )
        if src_fn is not None:
            su, sv = src_fn(self.Xc, self.Yc, t_now)
            dudt = dudt + su
            dvdt = dvdt + sv
        return dudt, dvdt

    def _dt_adv(self, u: np.ndarray, v: np.ndarray) -> float:
        umax = float(np.max(np.abs(u))) if u.size else 0.0
        vmax = float(np.max(np.abs(v))) if v.size else 0.0
        denom = 0.0
        if umax > 0.0:
            denom += umax / (self.dx + 1e-12)
        if vmax > 0.0:
            denom += vmax / (self.dy + 1e-12)
        if denom <= 0.0:
            return np.inf
        return self.CFL / (denom + 1e-12)

    def _dt_diff(self) -> float:
        if self.nu <= 0.0:
            return np.inf
        return self.CFL * min(
            0.25 * self.dx * self.dx / (self.nu + 1e-12),
            0.25 * self.dy * self.dy / (self.nu + 1e-12),
        )

    def solve(self) -> np.ndarray:
        dt_save = self.T / max(1, self.n_t)
        it_tot = int(np.ceil(self.T / dt_save)) + 1

        u, v = self._init_condition()
        source_fn = self._build_source()

        Ny, Nx = self.Ny, self.Nx
        out = np.zeros((it_tot, 2, Ny, Nx), dtype=float)
        t = 0.0
        tsave = 0.0
        i_save = 0
        out[i_save, 0] = u
        out[i_save, 1] = v

        use_fdm2 = (
            getattr(self, "advection_scheme", None) == "fdm2_euler_fixed"
        ) and self.nu <= 0.0
        while t < self.T - 1e-15:
            while t >= tsave - 1e-15 and i_save < it_tot - 1:
                if i_save > 0:
                    out[i_save, 0] = u
                    out[i_save, 1] = v
                i_save += 1
                tsave += dt_save

            if use_fdm2:
                dt_fixed = 1.0 / max(1.0, self.fixed_dt_hz)
                dt_cfl = self._dt_adv(u, v)
                dt = min(dt_fixed, dt_cfl, self.T - t, tsave - t)
            else:
                dt = min(self._dt_adv(u, v), self._dt_diff(), self.T - t, tsave - t)
            if dt <= 1e-12:
                break

            if (
                use_fdm2
                and source_fn is None
                and (
                    not self.bc_cfg
                    or str(self.bc_cfg.get("type", "periodic")).lower() == "periodic"
                )
            ):

                def upwind2_x(q, vel):
                    q_im1 = np.roll(q, 1, axis=1)
                    q_im2 = np.roll(q, 2, axis=1)
                    q_ip1 = np.roll(q, -1, axis=1)
                    q_ip2 = np.roll(q, -2, axis=1)
                    pos = vel >= 0.0
                    der_pos = (3.0 * q - 4.0 * q_im1 + q_im2) / (2.0 * self.dx)
                    der_neg = (-3.0 * q + 4.0 * q_ip1 - q_ip2) / (2.0 * self.dx)
                    return np.where(pos, der_pos, der_neg)

                def upwind2_y(q, vel):
                    q_jm1 = np.roll(q, 1, axis=0)
                    q_jm2 = np.roll(q, 2, axis=0)
                    q_jp1 = np.roll(q, -1, axis=0)
                    q_jp2 = np.roll(q, -2, axis=0)
                    pos = vel >= 0.0
                    der_pos = (3.0 * q - 4.0 * q_jm1 + q_jm2) / (2.0 * self.dy)
                    der_neg = (-3.0 * q + 4.0 * q_jp1 - q_jp2) / (2.0 * self.dy)
                    return np.where(pos, der_pos, der_neg)

                du_dx = upwind2_x(u, u)
                du_dy = upwind2_y(u, v)
                dv_dx = upwind2_x(v, u)
                dv_dy = upwind2_y(v, v)
                u = u - dt * (u * du_dx + v * du_dy)
                v = v - dt * (u * dv_dx + v * dv_dy)
            else:
                du1, dv1 = self._rhs(u, v, t, source_fn)
                u_mid = u + 0.5 * dt * du1
                v_mid = v + 0.5 * dt * dv1
                du2, dv2 = self._rhs(u_mid, v_mid, t + 0.5 * dt, source_fn)

                u = u + dt * du2
                v = v + dt * dv2
            t += dt

        out[-1, 0] = u
        out[-1, 1] = v
        return out


if __name__ == "__main__":
    eq = BurgersEquation2D(
        Lx=1.0,
        Ly=1.0,
        T=0.5,
        n_x=96,
        n_y=96,
        n_t=50,
        viscosity={"type": "constant", "value": 0.001},
        ic={
            "type": "gaussian",
            "A_u": 1.0,
            "A_v": 0.5,
            "sigma_x": 0.08,
            "sigma_y": 0.08,
            "rel_center_x": -0.2,
            "rel_center_y": -0.1,
        },
        source=None,
        CFL=0.5,
        second_order=True,
        bc={"type": "Periodic"},
    )

    print("Running solve with viscosity=0.001...")
    snapshots = eq.solve()
    print("Snapshots shape:", snapshots.shape)

    n_start = np.linalg.norm(snapshots[0, 0])
    n_end = np.linalg.norm(snapshots[-1, 0])
    print(f"t0 norm |u|: {n_start:.4f}")
    print(f"tN norm |u|: {n_end:.4f}")

    dmin, dmax = float(snapshots.min()), float(snapshots.max())
    print(f"Global min/max: {dmin:.4f}, {dmax:.4f}")

    if abs(dmax) > 100.0 or np.isnan(dmax):
        print("FAILURE: Simulation exploded.")
    else:
        print("SUCCESS: Simulation appears stable.")
