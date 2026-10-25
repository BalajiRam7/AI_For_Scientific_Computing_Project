import numpy as np
from typing import Dict, Optional, Callable


class NavierStokes2D:
    def __init__(
        self,
        Lx: float = 1.0,
        Ly: float = 1.0,
        T: float = 1.0,
        n_x: int = 64,
        n_y: int = 64,
        n_t: int = 100,
        viscosity: Optional[Dict] = None,
        ic: Optional[Dict] = None,
        source: Optional[Dict] = None,
        CFL: float = 0.5,
        dt: float | None = 1.0e-4,
        spectral_filter: Optional[Dict] = None,
    ) -> None:
        self.Lx, self.Ly, self.T = float(Lx), float(Ly), float(T)
        self.Nx, self.Ny, self.n_t = int(n_x), int(n_y), int(n_t)
        self.CFL = float(CFL)

        self.nu = float((viscosity or {}).get("value", 1e-3))
        self.ic_cfg = ic or {"type": "grf_operator"}
        self.source_cfg = source or {"type": "ns_table"}
        self.dt_fixed = float(dt if dt is not None else 1.0e-4)

        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        x = np.linspace(0.0, self.Lx, self.Nx, endpoint=False)
        y = np.linspace(0.0, self.Ly, self.Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y, indexing="xy")

        kx = 2.0 * np.pi * np.fft.fftfreq(self.Nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.Ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="xy")
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1.0  # avoid /0 in Poisson
        self.kmag = np.sqrt(self.KX**2 + self.KY**2)

        sf = spectral_filter or {}
        self.filter_enabled = bool(sf.get("enabled", False))
        self.filter_alpha = float(sf.get("alpha", 0.02))
        self.filter_p = int(sf.get("p", 8))
        self.filter_kcut = sf.get("kcut", None)

        self._src_fn = self._build_source(self.source_cfg)

    def _init_vorticity(self) -> np.ndarray:
        kind = str(self.ic_cfg.get("type", "random")).lower()

        if kind == "taylor-green":
            k0 = int(self.ic_cfg.get("k", 4))
            U0 = float(self.ic_cfg.get("U0", 1.0))
            omega = 2.0 * U0 * k0 * np.sin(k0 * self.X) * np.sin(k0 * self.Y)

        elif kind == "gaussian":
            cx = float(self.ic_cfg.get("cx", self.Lx / 2))
            cy = float(self.ic_cfg.get("cy", self.Ly / 2))
            sigma = float(self.ic_cfg.get("sigma", 0.3))
            omega = np.exp(
                -((self.X - cx) ** 2 + (self.Y - cy) ** 2) / (2.0 * sigma**2)
            )

        elif kind == "random":
            seed_val = self.ic_cfg.get("seed", None)
            if seed_val is None or (
                isinstance(seed_val, str)
                and seed_val.strip().lower() in ("", "none", "off", "false")
            ):
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(int(seed_val))
            spectrum = rng.standard_normal(
                (self.Ny, self.Nx)
            ) + 1j * rng.standard_normal((self.Ny, self.Nx))
            filt = np.exp(-0.02 * (self.K2))
            omega = np.fft.ifft2(spectrum * filt).real
            omega -= np.mean(omega)
            omega /= np.std(omega) + 1e-12

        elif kind in ("grf_operator", "grf_matern"):
            c = float(self.ic_cfg.get("scale", 73.0 / 2.0))
            seed_val = self.ic_cfg.get("seed", None)
            if seed_val is None or (
                isinstance(seed_val, str)
                and seed_val.strip().lower() in ("", "none", "off", "false")
            ):
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(int(seed_val))
            xi = rng.standard_normal((self.Ny, self.Nx))
            xi_hat = np.fft.fft2(xi)
            S = c / (np.power(self.K2 + 49.0, 2.5))
            S[0, 0] = 0.0
            omega_hat = np.sqrt(S) * xi_hat
            omega = np.fft.ifft2(omega_hat).real
            omega -= np.mean(omega)
            if bool(self.ic_cfg.get("normalize", True)):
                std = float(np.std(omega) + 1e-12)
                omega = omega / std

        elif kind == "multi-blob":
            seed_val = self.ic_cfg.get("seed", None)
            if seed_val is None or (
                isinstance(seed_val, str)
                and seed_val.strip().lower() in ("", "none", "off", "false")
            ):
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(int(seed_val))
            n = int(self.ic_cfg.get("n", 24))
            sigma = float(self.ic_cfg.get("sigma", 0.08))
            A = float(self.ic_cfg.get("A", 1.0))
            omega = np.zeros((self.Ny, self.Nx), dtype=float)
            for _ in range(n):
                cx = rng.uniform(0.0, self.Lx)
                cy = rng.uniform(0.0, self.Ly)
                sgn = rng.choice([-1.0, 1.0])
                blob = np.exp(
                    -((self.X - cx) ** 2 + (self.Y - cy) ** 2) / (2.0 * sigma**2)
                )
                omega += sgn * A * blob
            omega -= np.mean(omega)
            omega /= np.std(omega) + 1e-12

        elif kind == "band-limited":
            seed_val = self.ic_cfg.get("seed", None)
            kmin = float(self.ic_cfg.get("kmin", 4.0))
            kmax = float(self.ic_cfg.get("kmax", 10.0))
            if seed_val is None or (
                isinstance(seed_val, str)
                and seed_val.strip().lower() in ("", "none", "off", "false")
            ):
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(int(seed_val))
            phase = rng.uniform(0, 2 * np.pi, size=(self.Ny, self.Nx))
            spec = np.zeros((self.Ny, self.Nx), dtype=complex)
            mask = (self.kmag >= kmin) & (self.kmag <= kmax)
            spec[mask] = np.cos(phase[mask]) + 1j * np.sin(phase[mask])
            omega = np.fft.ifft2(spec).real
            omega -= np.mean(omega)
            omega /= np.std(omega) + 1e-12

        else:
            raise ValueError(f"Unknown IC type: {kind}")

        omega -= np.mean(omega)
        return omega

    @staticmethod
    def _kolmogorov_source_factory(
        f0: float,
        kf: int,
        axis: str,
        Lx: float,
        Ly: float,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> Callable[[float], np.ndarray]:
        """Vorticity source curl(f)_z. axis 'y': -f0 k cos(k y); axis 'x': +f0 k cos(k x)."""
        axis = axis.lower()
        if axis == "y":
            ky = kf * 2.0 * np.pi / Ly
            return lambda t: -f0 * ky * np.cos(ky * Y)
        else:
            kx = kf * 2.0 * np.pi / Lx
            return lambda t: +f0 * kx * np.cos(kx * X)

    def _build_source(
        self, src_cfg: Optional[Dict]
    ) -> Optional[Callable[[float], np.ndarray]]:
        if not src_cfg:
            return None
        kind = str(src_cfg.get("type", "none")).lower()
        if kind == "ns_table":
            return lambda t: 0.1 * (
                np.sin(2.0 * np.pi * (self.X + self.Y))
                + np.cos(2.0 * np.pi * (self.X + self.Y))
            )
        if kind == "kolmogorov":
            f0 = float(src_cfg.get("f0", 0.03))
            kf = int(src_cfg.get("kf", 2))
            axis = str(src_cfg.get("axis", "y"))
            return self._kolmogorov_source_factory(
                f0, kf, axis, self.Lx, self.Ly, self.X, self.Y
            )
        return None

    def _poisson_streamfunction(self, omega: np.ndarray) -> np.ndarray:
        omega_hat = np.fft.fft2(omega)
        psi_hat = -omega_hat / self.K2
        psi_hat[0, 0] = 0.0
        return np.fft.ifft2(psi_hat).real

    def _velocity_from_psi(self, psi: np.ndarray):
        u = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0 * self.dy)
        v = -(np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0 * self.dx)
        return u, v

    def _spec_velocity_from_omega(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        omega_hat = np.fft.fft2(omega)
        psi_hat = -omega_hat / self.K2
        psi_hat[0, 0] = 0.0
        u = np.fft.ifft2(1j * self.KY * psi_hat).real
        v = np.fft.ifft2(-1j * self.KX * psi_hat).real
        return u, v

    def _advection_term(self, omega: np.ndarray) -> np.ndarray:
        omega_hat = np.fft.fft2(omega)
        omegax = np.fft.ifft2(1j * self.KX * omega_hat).real
        omegay = np.fft.ifft2(1j * self.KY * omega_hat).real
        u, v = self._spec_velocity_from_omega(omega)
        return u * omegax + v * omegay

    def _laplacian(self, q: np.ndarray):
        return (
            np.roll(q, -1, axis=1) - 2.0 * q + np.roll(q, 1, axis=1)
        ) / self.dx**2 + (
            np.roll(q, -1, axis=0) - 2.0 * q + np.roll(q, 1, axis=0)
        ) / self.dy**2

    def _arakawa_J(self, psi: np.ndarray, omega: np.ndarray) -> np.ndarray:
        dx, dy = self.dx, self.dy
        dA = dx * dy

        psi_ip = np.roll(psi, -1, axis=1)
        psi_im = np.roll(psi, 1, axis=1)
        psi_jp = np.roll(psi, -1, axis=0)
        psi_jm = np.roll(psi, 1, axis=0)

        w_ip = np.roll(omega, -1, axis=1)
        w_im = np.roll(omega, 1, axis=1)
        w_jp = np.roll(omega, -1, axis=0)
        w_jm = np.roll(omega, 1, axis=0)

        psi_ip_jp = np.roll(psi_jp, -1, axis=1)
        psi_im_jp = np.roll(psi_jp, 1, axis=1)
        psi_ip_jm = np.roll(psi_jm, -1, axis=1)
        psi_im_jm = np.roll(psi_jm, 1, axis=1)

        w_ip_jp = np.roll(w_jp, -1, axis=1)
        w_im_jp = np.roll(w_jp, 1, axis=1)
        w_ip_jm = np.roll(w_jm, -1, axis=1)
        w_im_jm = np.roll(w_jm, 1, axis=1)

        J1 = ((psi_ip - psi_im) * (w_jp - w_jm) - (psi_jp - psi_jm) * (w_ip - w_im)) / (
            4.0 * dx * dy
        )

        J2 = (
            psi_ip * (w_ip_jp - w_ip_jm)
            - psi_im * (w_im_jp - w_im_jm)
            - psi_jp * (w_ip_jp - w_im_jp)
            + psi_jm * (w_ip_jm - w_im_jm)
        ) / (4.0 * dA)

        J3 = (
            psi_ip_jp * (w_jp - w_ip)
            - psi_im_jp * (w_jp - w_im)
            - psi_ip_jm * (w_jm - w_ip)
            + psi_im_jm * (w_jm - w_im)
        ) / (4.0 * dA)

        return (J1 + J2 + J3) / 3.0

    def _apply_spectral_filter(self, omega: np.ndarray) -> np.ndarray:
        if not self.filter_enabled:
            return omega

        omega_hat = np.fft.fft2(omega)

        kx_max = np.max(np.abs(self.KX))
        ky_max = np.max(np.abs(self.KY))
        kx_cut = (2.0 / 3.0) * kx_max
        ky_cut = (2.0 / 3.0) * ky_max
        mask = (np.abs(self.KX) <= (self.filter_kcut or kx_cut)) & (
            np.abs(self.KY) <= (self.filter_kcut or ky_cut)
        )
        omega_hat *= mask

        kmax = np.max(self.kmag)
        r = self.kmag / (kmax + 1e-12)
        roll = np.exp(-self.filter_alpha * (np.clip(r, 0, 1) ** self.filter_p))
        omega_hat *= roll

        return np.fft.ifft2(omega_hat).real

    def _rhs(self, omega: np.ndarray, t: float) -> np.ndarray:
        omega = omega - np.mean(omega)

        psi = self._poisson_streamfunction(omega)
        u, v = self._velocity_from_psi(psi)

        J = self._arakawa_J(psi, omega)
        rhs = -J + self.nu * self._laplacian(omega)

        if self._src_fn is not None:
            rhs = rhs + self._src_fn(t)

        return rhs

    def solve(self) -> Dict[str, np.ndarray]:
        omega = self._init_vorticity()

        dt = self.dt_fixed
        dt_save = self.T / max(1, self.n_t)
        n_save = int(np.ceil(self.T / dt_save)) + 1

        snaps_omega = np.zeros((n_save, self.Ny, self.Nx), dtype=float)
        snaps_u = np.zeros_like(snaps_omega)
        snaps_v = np.zeros_like(snaps_omega)

        u, v = self._spec_velocity_from_omega(omega)
        snaps_omega[0], snaps_u[0], snaps_v[0] = omega, u, v

        a = 1.0 + 0.5 * dt * self.nu * self.K2
        b = 1.0 - 0.5 * dt * self.nu * self.K2

        t, i_save, t_next_save = 0.0, 1, dt_save
        N_prev = None

        while t < self.T - 1e-15:
            N = self._advection_term(omega)

            f_now = self._src_fn(t + 0.5 * dt) if self._src_fn is not None else 0.0
            NL = None
            if N_prev is None:
                NL = -N + f_now
            else:
                NL = -(1.5 * N - 0.5 * N_prev) + f_now

            omega_hat = np.fft.fft2(omega)
            NL_hat = np.fft.fft2(NL) if not np.isscalar(NL) else 0.0
            rhs_hat = b * omega_hat + dt * NL_hat
            omega_hat_new = rhs_hat / a
            omega = np.fft.ifft2(omega_hat_new).real

            omega = self._apply_spectral_filter(omega)

            t += dt
            u, v = self._spec_velocity_from_omega(omega)
            N_prev = N

            while (t >= t_next_save - 1e-12) and (i_save < n_save):
                snaps_omega[i_save] = omega
                snaps_u[i_save] = u
                snaps_v[i_save] = v
                i_save += 1
                t_next_save += dt_save

        snaps_omega[-1] = omega
        snaps_u[-1] = u
        snaps_v[-1] = v
        return snaps_omega


if __name__ == "__main__":
    eq = NavierStokes2D(
        Lx=2 * np.pi,
        Ly=2 * np.pi,
        T=20.0,
        n_x=80,
        n_y=80,
        n_t=100,
        viscosity={"type": "constant", "value": 8e-4},
        ic={"type": "band-limited", "kmin": 4, "kmax": 10, "seed": 7},
        source={"type": "kolmogorov", "f0": 0.03, "kf": 2, "axis": "y"},
        CFL=0.5,
        spectral_filter={
            "enabled": True,
            "alpha": 0.02,
            "p": 8,
        },
    )

    res = eq.solve()
    print(res.shape)
