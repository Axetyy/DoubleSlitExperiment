"""Unified physics implementations for 2D and 3D simulation backends.

CPU-optimised version: the whole time step (stencil, damping, wall mask,
source injection, detector averaging) is fused into one parallel Numba
kernel.
"""

import os
import ctypes
import shlex
import shutil
import subprocess
import sys

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
PARALLEL = os.environ.get("WAVESIM_PARALLEL", "1") != "0"

import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return lambda function: function

from config import Params

try:
    import torch
    HAVE_TORCH = True
except Exception:
    torch = None
    HAVE_TORCH = False


@njit(parallel=PARALLEL, cache=False)
def _one_step(u, up, un, c_next, c_prev, C2, sy, sx, sv, x0, x1, I_det, avg):
    """One leapfrog step, fused.

    un = mask * ( dn * (2u + C2*lap(u)) - dn*dp * up + source )
    with c_next = dn*mask and c_prev = dn*dp*mask precomputed.
    Border cells have lap == 0
    """
    ny, nx = u.shape
    two = np.float32(2.0)
    four = np.float32(4.0)
    inv_w = 1.0 / (x1 - x0 + 1)

    for i in prange(ny):
        if i == 0 or i == ny - 1:
            for j in range(nx):
                un[i, j] = c_next[i, j] * (two * u[i, j]) - c_prev[i, j] * up[i, j]
        else:
            un[i, 0] = c_next[i, 0] * (two * u[i, 0]) - c_prev[i, 0] * up[i, 0]
            un[i, nx - 1] = (
                c_next[i, nx - 1] * (two * u[i, nx - 1])
                - c_prev[i, nx - 1] * up[i, nx - 1]
            )
            for j in range(1, nx - 1):
                lap = C2 * (
                    u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1]
                    - four * u[i, j]
                )
                un[i, j] = c_next[i, j] * (two * u[i, j] + lap) - c_prev[i, j] * up[i, j]

        if i == sy:
            un[i, sx] += sv

        # detector: running mean of u^2 over the last columns, float64 accumulator
        s = 0.0
        for j in range(x0, x1 + 1):
            v = un[i, j]
            s += v * v
        I_det[i] += np.float32((s * inv_w - I_det[i]) / avg)


@njit(cache=False)
def _run_steps(u, up, un, c_next, c_prev, C2, sy, sx, sv_arr, x0, x1, I_det, avg):
    """Run len(sv_arr) steps, rotating the three buffers. Returns (u, u_prev, u_next)."""
    for k in range(sv_arr.shape[0]):
        _one_step(u, up, un, c_next, c_prev, C2, sy, sx, sv_arr[k], x0, x1, I_det, avg)
        u, up, un = un, u, up
    return u, up, un


def _warmup():
    """Trigger JIT compilation on a tiny dummy problem."""
    a = np.zeros((8, 8), np.float32)
    b = np.zeros_like(a)
    c = np.zeros_like(a)
    ones = np.ones_like(a)
    I = np.zeros(8, np.float32)
    _run_steps(a, b, c, ones, ones, np.float32(0.25), 4, 2,
               np.zeros(1, np.float32), 4, 6, I, 480.0)


_WARMED = False


class WaveSimulator:
    AVG_STEPS = 480

    def __init__(self, p: Params):
        global _WARMED
        if not HAVE_NUMBA:
            raise RuntimeError("numba is not installed (pip install numba).")
        if not _WARMED:
            _warmup()
            _WARMED = True

        self.p = p

        shape = (p.ny, p.nx)

        self.u = np.zeros(shape, np.float32)
        self.u_prev = np.zeros_like(self.u)
        self.u_next = np.zeros_like(self.u)

        self.I_det = np.zeros(p.ny, np.float32)

        self.damping = self._build_absorber()
        self.damping_prev = (1 - 0.5 * self.damping).astype(np.float32)
        self.damping_next = (1 / (1 + 0.5 * self.damping)).astype(np.float32)

        self.rng = np.random.default_rng()

        self.t_step = 0

        self.rebuild(fresh=True)

    def _build_absorber(self):
        p = self.p

        x = np.arange(p.nx)
        y = np.arange(p.ny)

        depth_x = np.maximum(0, p.sponge - np.minimum(x, p.nx - 1 - x)) / p.sponge
        depth_y = np.maximum(0, p.sponge - np.minimum(y, p.ny - 1 - y)) / p.sponge
        edge_depth = np.maximum(depth_y[:, None], depth_x[None, :])
        return (0.6 * edge_depth ** 2).astype(np.float32)

    def rebuild(self, fresh=False):
        p = self.p
        yc = p.ny / 2
        y = np.arange(p.ny)

        self.slit_centers = (
            yc + (np.arange(p.n_slits) - (p.n_slits - 1) / 2) * p.slit_spacing
        )

        self.slit_open = np.zeros(p.ny, dtype=bool)

        for c in self.slit_centers:
            self.slit_open |= np.abs(y - c) <= p.slit_width / 2

        open_ = np.ones((p.ny, p.nx), dtype=np.float32)
        open_[:, p.x_wall:p.x_wall + p.wall_thickness] = self.slit_open[:, None]

        self.wall_mask = open_

        tan_t = np.tan(np.deg2rad(p.source_angle_deg))
        yc_src = np.clip(
            yc - (p.x_wall - p.x_src) * tan_t,
            p.sponge,
            p.ny - p.sponge - 1,
        )
        self.src_y = np.array([int(round(yc_src))])
        k = 2 * np.pi / p.wavelength

        self.phase_offset = (
            -k * (self.src_y - yc_src) * np.sin(np.deg2rad(p.source_angle_deg))
        )
        self.phase_noise = np.zeros(len(self.src_y))
        self.omega = 2 * np.pi / p.wavelength

        self.C2 = np.float32(p.courant ** 2)
        
        self.c_next = np.ascontiguousarray(
            self.damping_next * self.wall_mask, dtype=np.float32
        )
        self.c_prev = np.ascontiguousarray(
            self.damping_prev * self.damping_next * self.wall_mask, dtype=np.float32
        )
        self.src_mask = np.float32(self.wall_mask[self.src_y[0], p.x_src])

        propagation_steps = (p.x_det - p.x_src) / p.courant

        self.settle = int(propagation_steps + 2 * self.AVG_STEPS)

        if fresh:
            self.settle += int((p.x_wall - p.x_src) / p.courant)

    def reset(self):
        self.u.fill(0)
        self.u_prev.fill(0)
        self.u_next.fill(0)
        self.I_det.fill(0)

        self.t_step = 0

        self.rebuild(fresh=True)

    def step_n(self, n=1):
        """Advance n time steps in a single compiled call."""
        if n <= 0:
            return

        p = self.p

        t = self.t_step + np.arange(n)
        phase = self.omega * p.courant * t + self.phase_offset[0]

        if not p.coherent:
            noise = self.phase_noise[0] + np.cumsum(0.1 * self.rng.standard_normal(n))
            self.phase_noise[0] = noise[-1]
            phase = phase + noise

        amplitude = np.minimum(1.0, t / 100)
        sv = (amplitude * np.sin(phase) * self.src_mask).astype(np.float32)

        x0 = max(0, p.x_det - p.detector_width + 1)
        
        ny, nx = self.u.shape
        if not (0 <= int(self.src_y[0]) < ny and 0 <= p.x_src < nx and 0 <= p.x_det < nx):
            raise ValueError(
                f"index outside grid: src=({self.src_y[0]},{p.x_src}) x_det={p.x_det} grid=({ny},{nx})"
            )
        for a in (self.u, self.u_prev, self.u_next, self.c_next, self.c_prev):
            if a.shape != (ny, nx) or a.dtype != np.float32 or not a.flags.c_contiguous:
                raise ValueError("simulation arrays must be C-contiguous float32 of identical shape")

        self.u, self.u_prev, self.u_next = _run_steps(
            self.u, self.u_prev, self.u_next,
            self.c_next, self.c_prev, self.C2,
            int(self.src_y[0]), int(p.x_src), sv,
            int(x0), int(p.x_det),
            self.I_det, float(self.AVG_STEPS),
        )

        self.t_step += n
        self.settle -= n

    def step(self):
        self.step_n(1)
    def detector_image(self):
        return self.I_det


def cuda_available():
    return HAVE_TORCH and torch.cuda.is_available()


def pick_device(name=None):
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch is not installed (pip install torch).")
    if name in (None, "cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_label(device):
    if device.type == "cuda":
        return f"GPU torch ({torch.cuda.get_device_name(device)})"
    return "CPU torch"


class TorchWaveSim2D:
    AVG_STEPS = 480

    def __init__(self, p, device=None):
        self.p = p
        self.device = pick_device(device)
        self.label = device_label(self.device)
        self.rng = np.random.default_rng()
        self.t_step = 0
        shape = (p.ny, p.nx)
        self.u = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.u_prev = torch.zeros_like(self.u)
        self.u_next = torch.zeros_like(self.u)
        self.I_det = torch.zeros(p.ny, dtype=torch.float32, device=self.device)
        self.damping = self._build_absorber()
        self.damping_prev = (1 - 0.5 * self.damping).astype(np.float32)
        self.damping_next = (1 / (1 + 0.5 * self.damping)).astype(np.float32)
        self.rebuild(fresh=True)

    def _build_absorber(self):
        p = self.p
        x = np.arange(p.nx)
        y = np.arange(p.ny)
        depth_x = np.maximum(0, p.sponge - np.minimum(x, p.nx - 1 - x)) / p.sponge
        depth_y = np.maximum(0, p.sponge - np.minimum(y, p.ny - 1 - y)) / p.sponge
        edge_depth = np.maximum(depth_y[:, None], depth_x[None, :])
        return (0.6 * edge_depth ** 2).astype(np.float32)

    def rebuild(self, fresh=False):
        p = self.p
        yc = p.ny / 2
        y = np.arange(p.ny)
        self.slit_centers = (
            yc + (np.arange(p.n_slits) - (p.n_slits - 1) / 2) * p.slit_spacing
        )
        self.slit_open = np.zeros(p.ny, dtype=bool)
        for center in self.slit_centers:
            self.slit_open |= np.abs(y - center) <= p.slit_width / 2
        self.wall_mask = np.ones((p.ny, p.nx), dtype=np.float32)
        self.wall_mask[:, p.x_wall:p.x_wall + p.wall_thickness] = self.slit_open[:, None]
        tan_t = np.tan(np.deg2rad(p.source_angle_deg))
        source_y = np.clip(
            yc - (p.x_wall - p.x_src) * tan_t,
            p.sponge,
            p.ny - p.sponge - 1,
        )
        self.src_y = np.array([int(round(source_y))])
        k = 2 * np.pi / p.wavelength
        self.phase_offset = (
            -k * (self.src_y - source_y) * np.sin(np.deg2rad(p.source_angle_deg))
        )
        self.phase_noise = np.zeros(len(self.src_y))
        self.omega = 2 * np.pi / p.wavelength
        self.C2 = np.float32(p.courant ** 2)
        self.c_next = torch.from_numpy(np.ascontiguousarray(
            self.damping_next * self.wall_mask, dtype=np.float32
        )).to(self.device)
        self.c_prev = torch.from_numpy(np.ascontiguousarray(
            self.damping_prev * self.damping_next * self.wall_mask, dtype=np.float32
        )).to(self.device)
        self.src_mask = np.float32(self.wall_mask[self.src_y[0], p.x_src])
        propagation_steps = (p.x_det - p.x_src) / p.courant
        self.settle = int(propagation_steps + 2 * self.AVG_STEPS)
        if fresh:
            self.settle += int((p.x_wall - p.x_src) / p.courant)

    def reset(self):
        self.u.zero_()
        self.u_prev.zero_()
        self.u_next.zero_()
        self.I_det.zero_()
        self.t_step = 0
        self.rebuild(fresh=True)

    def _source_values(self, n):
        p = self.p
        t = self.t_step + np.arange(n)
        phase = self.omega * p.courant * t + self.phase_offset[0]
        if not p.coherent:
            noise = self.phase_noise[0] + np.cumsum(0.1 * self.rng.standard_normal(n))
            self.phase_noise[0] = noise[-1]
            phase = phase + noise
        amplitude = np.minimum(1.0, t / 100)
        return (amplitude * np.sin(phase) * self.src_mask).astype(np.float32)

    def step_n(self, n=1):
        if n <= 0:
            return 0
        p = self.p
        source_values = torch.from_numpy(self._source_values(n)).to(self.device)
        u, u_prev, u_next = self.u, self.u_prev, self.u_next
        x0 = max(0, p.x_det - p.detector_width + 1)
        C2 = float(p.courant ** 2)
        sy, sx = int(self.src_y[0]), int(p.x_src)
        inv_avg = 1.0 / self.AVG_STEPS
        with torch.no_grad():
            for k in range(n):
                torch.mul(u, 2.0, out=u_next)
                lap = u[2:, 1:-1] + u[:-2, 1:-1]
                lap += u[1:-1, 2:]
                lap += u[1:-1, :-2]
                lap.sub_(u[1:-1, 1:-1], alpha=4.0)
                u_next[1:-1, 1:-1].add_(lap, alpha=C2)
                u_next.mul_(self.c_next)
                u_next.addcmul_(self.c_prev, u_prev, value=-1.0)
                u_next[sy, sx].add_(source_values[k])
                instance = u_next[:, x0:p.x_det + 1].pow(2).mean(dim=1)
                self.I_det += (instance - self.I_det) * inv_avg
                u, u_prev, u_next = u_next, u, u_prev
        self.u, self.u_prev, self.u_next = u, u_prev, u_next
        self.t_step += n
        self.settle -= n

    def step(self):
        self.step_n(1)

    def field_slice(self):
        return self.u.detach().cpu().numpy()

    def detector_profile(self):
        return self.I_det.detach().cpu().numpy()

    def detector_image(self):
        return self.detector_profile()


@njit(parallel=PARALLEL, cache=False)
def _one_step_3d(u, up, un, c_next, c_prev, C2, sz, sy, sx, sv, x0, x1, I_det, avg):
    nz, ny, nx = u.shape
    two = np.float32(2.0)
    six = np.float32(6.0)
    inv_w = 1.0 / (x1 - x0 + 1)
    for k in prange(nz):
        for i in range(ny):
            if k == 0 or k == nz - 1 or i == 0 or i == ny - 1:
                for j in range(nx):
                    un[k, i, j] = c_next[k, i, j] * (two * u[k, i, j]) - c_prev[k, i, j] * up[k, i, j]
            else:
                un[k, i, 0] = c_next[k, i, 0] * (two * u[k, i, 0]) - c_prev[k, i, 0] * up[k, i, 0]
                un[k, i, nx - 1] = (
                    c_next[k, i, nx - 1] * (two * u[k, i, nx - 1])
                    - c_prev[k, i, nx - 1] * up[k, i, nx - 1]
                )
                for j in range(1, nx - 1):
                    lap = C2 * (
                        u[k + 1, i, j] + u[k - 1, i, j]
                        + u[k, i + 1, j] + u[k, i - 1, j]
                        + u[k, i, j + 1] + u[k, i, j - 1]
                        - six * u[k, i, j]
                    )
                    un[k, i, j] = c_next[k, i, j] * (two * u[k, i, j] + lap) - c_prev[k, i, j] * up[k, i, j]
            if k == sz and i == sy:
                un[k, i, sx] += sv
            s = 0.0
            for j in range(x0, x1 + 1):
                value = un[k, i, j]
                s += value * value
            I_det[i, k] += np.float32((s * inv_w - I_det[i, k]) / avg)


@njit(cache=False)
def _run_steps_3d(u, up, un, cn, cp, C2, sz, sy, sx, sv_arr, x0, x1, I_det, avg):
    for n in range(sv_arr.shape[0]):
        _one_step_3d(u, up, un, cn, cp, C2, sz, sy, sx, sv_arr[n], x0, x1, I_det, avg)
        u, up, un = un, u, up
    return u, up, un


_WARMED_3D = False


def _warmup_3d():
    global _WARMED_3D
    if _WARMED_3D:
        return
    a = np.zeros((6, 6, 8), np.float32)
    b = np.zeros_like(a)
    c = np.zeros_like(a)
    ones = np.ones_like(a)
    intensity = np.zeros((6, 6), np.float32)
    _run_steps_3d(a, b, c, ones, ones, np.float32(0.25), 3, 3, 2,
                  np.zeros(1, np.float32), 4, 6, intensity, 480.0)
    _WARMED_3D = True


class _Base3d:
    AVG_STEPS = 480
    label = ""

    def __init__(self, p):
        self.p = p
        self.rng = np.random.default_rng()
        self.t_step = 0
        self.nz, self.ny, self.nx = p.nz, p.ny, p.nx
        self._build_damping()
        self._alloc_state()
        self.rebuild(fresh=True)

    def _build_damping(self):
        p = self.p

        def depth(n):
            i = np.arange(n)
            return (np.maximum(0, p.sponge - np.minimum(i, n - 1 - i)) / p.sponge).astype(np.float32)

        dz = depth(p.nz)[:, None, None]
        dy = depth(p.ny)[None, :, None]
        dx = depth(p.nx)[None, None, :]
        damping = (0.6 * np.maximum(np.maximum(dz, dy), dx) ** 2).astype(np.float32)
        self.dprev = (1 - 0.5 * damping).astype(np.float32)
        self.dnext = (1 / (1 + 0.5 * damping)).astype(np.float32)

    def rebuild(self, fresh=False):
        p = self.p
        yc, zc = p.ny / 2, p.nz / 2
        y, z = np.arange(p.ny), np.arange(p.nz)
        self.slit_centers = (
            yc + (np.arange(p.n_slits) - (p.n_slits - 1) / 2) * p.slit_spacing
        )
        open_y = np.zeros(p.ny, dtype=bool)
        for center in self.slit_centers:
            open_y |= np.abs(y - center) <= p.slit_width / 2
        open_z = np.abs(z - zc) <= p.slit_height / 2
        self.slit_open = open_y
        self.slit_open_z = open_z
        aperture = open_z[:, None] & open_y[None, :]
        if p.slit_shape == "circular":
            aperture = np.zeros((p.nz, p.ny), dtype=bool)
            radius_squared = (p.slit_width / 2) ** 2
            for center in self.slit_centers:
                aperture |= (
                    (z[:, None] - zc) ** 2 + (y[None, :] - center) ** 2
                    <= radius_squared
                )
        self.aperture_mask = aperture
        mask = np.ones((p.nz, p.ny, p.nx), dtype=np.float32)
        mask[:, :, p.x_wall:p.x_wall + p.wall_thickness] = aperture[:, :, None]
        angle = np.tan(np.deg2rad(p.source_angle_deg))
        source_y = np.clip(
            yc - (p.x_wall - p.x_src) * angle,
            p.sponge,
            p.ny - p.sponge - 1,
        )
        self.src_y = np.array([int(round(source_y))])
        self.src_z = int(p.nz // 2)
        k = 2 * np.pi / p.wavelength
        self.phase_offset = (
            -k * (self.src_y - source_y) * np.sin(np.deg2rad(p.source_angle_deg))
        )
        self.phase_noise = 0.0
        self.omega = 2 * np.pi / p.wavelength
        self.src_mask = float(mask[self.src_z, self.src_y[0], p.x_src])
        self._upload_coeffs(self.dnext * mask, self.dprev * self.dnext * mask)
        self.settle = int((p.x_det - p.x_src) / p.courant + 2 * self.AVG_STEPS)
        if fresh:
            self.settle += int((p.x_wall - p.x_src) / p.courant)

    def reset(self):
        self._zero_state()
        self.t_step = 0
        self.rebuild(fresh=True)

    def _source_values(self, n):
        p = self.p
        t = self.t_step + np.arange(n)
        phase = self.omega * p.courant * t + self.phase_offset
        if not p.coherent:
            noise = self.phase_noise + np.cumsum(0.1 * self.rng.standard_normal(n))
            self.phase_noise = float(noise[-1])
            phase = phase + noise
        amplitude = np.minimum(1.0, t / 100)
        return (amplitude * np.sin(phase) * self.src_mask).astype(np.float32)

    def step_n(self, n=1):
        if n <= 0:
            return 0
        self._advance(self._source_values(n))
        self.t_step += n
        self.settle -= n

    def step(self):
        self.step_n(1)


class WaveSim3D(_Base3d):
    label = "CPU numba"

    def __init__(self, p, device=None):
        if not HAVE_NUMBA:
            raise RuntimeError("numba is not installed (pip install numba).")
        _warmup_3d()
        super().__init__(p)

    def _alloc_state(self):
        shape = (self.nz, self.ny, self.nx)
        self.u = np.zeros(shape, np.float32)
        self.u_prev = np.zeros_like(self.u)
        self.u_next = np.zeros_like(self.u)
        self.I_det = np.zeros((self.ny, self.nz), np.float32)

    def _zero_state(self):
        self.u.fill(0)
        self.u_prev.fill(0)
        self.u_next.fill(0)
        self.I_det.fill(0)

    def _upload_coeffs(self, cn, cp):
        self.cn = np.ascontiguousarray(cn, dtype=np.float32)
        self.cp = np.ascontiguousarray(cp, dtype=np.float32)

    def _advance(self, sv):
        p = self.p
        if not (0 <= self.src_y[0] < self.ny and 0 <= p.x_src < self.nx and 0 <= p.x_det < self.nx):
            raise ValueError("source or detector outside the grid")
        x0 = max(0, p.x_det - p.detector_width + 1)
        self.u, self.u_prev, self.u_next = _run_steps_3d(
            self.u, self.u_prev, self.u_next, self.cn, self.cp,
            np.float32(p.courant ** 2),
            int(self.src_z), int(self.src_y[0]), int(p.x_src), sv,
            int(x0), int(p.x_det), self.I_det, float(self.AVG_STEPS),
        )

    def field_slice(self):
        return self.u[self.nz // 2]

    def detector_image(self):
        return self.I_det

    def detector_profile(self):
        return self.I_det[:, self.nz // 2]


class TorchWaveSim3D(_Base3d):
    def __init__(self, p, device=None):
        if not HAVE_TORCH:
            raise RuntimeError("PyTorch is not installed (pip install torch).")
        if device in (None, "cuda", "gpu") and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.label = f"GPU torch ({torch.cuda.get_device_name(self.device)})"
        else:
            self.device = torch.device("cpu")
            self.label = "CPU torch"
        super().__init__(p)

    def _alloc_state(self):
        shape = (self.nz, self.ny, self.nx)
        self.u = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.u_prev = torch.zeros_like(self.u)
        self.u_next = torch.zeros_like(self.u)
        self.I_det = torch.zeros((self.ny, self.nz), dtype=torch.float32, device=self.device)

    def _zero_state(self):
        self.u.zero_()
        self.u_prev.zero_()
        self.u_next.zero_()
        self.I_det.zero_()

    def _upload_coeffs(self, cn, cp):
        self.cn = torch.from_numpy(np.ascontiguousarray(cn, dtype=np.float32)).to(self.device)
        self.cp = torch.from_numpy(np.ascontiguousarray(cp, dtype=np.float32)).to(self.device)

    def _advance(self, sv_np):
        p = self.p
        sv = torch.from_numpy(sv_np).to(self.device)
        u, up, un = self.u, self.u_prev, self.u_next
        C2 = float(p.courant ** 2)
        sz, sy, sx = int(self.src_z), int(self.src_y[0]), int(p.x_src)
        x0, x1 = max(0, p.x_det - p.detector_width + 1), p.x_det
        inv_avg = 1.0 / self.AVG_STEPS
        with torch.no_grad():
            for n in range(sv.shape[0]):
                torch.mul(u, 2.0, out=un)
                lap = u[2:, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]
                lap += u[1:-1, 2:, 1:-1]
                lap += u[1:-1, :-2, 1:-1]
                lap += u[1:-1, 1:-1, 2:]
                lap += u[1:-1, 1:-1, :-2]
                lap.sub_(u[1:-1, 1:-1, 1:-1], alpha=6.0)
                un[1:-1, 1:-1, 1:-1].add_(lap, alpha=C2)
                un.mul_(self.cn)
                un.addcmul_(self.cp, up, value=-1.0)
                un[sz, sy, sx].add_(sv[n])
                instance = un[:, :, x0:x1 + 1].pow(2).mean(dim=2)
                self.I_det += (instance.T - self.I_det) * inv_avg
                u, up, un = un, u, up
        self.u, self.u_prev, self.u_next = u, up, un

    def field_slice(self):
        return self.u[self.nz // 2].detach().cpu().numpy()

    def detector_image(self):
        return self.I_det.detach().cpu().numpy()

    def detector_profile(self):
        return self.I_det[:, self.nz // 2].detach().cpu().numpy()


_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE_DIR = os.path.join(_HERE, "C-port")
_EXTENSION = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
_LIBRARY_PATH = os.path.join(_SOURCE_DIR, "libphysics" + _EXTENSION)
_LIBRARY = None
_DLL_DIRECTORY = None
_PARAM_NAMES = (
    "x_src", "x_wall", "x_det", "nx", "ny", "wall_thickness", "sponge", "n_slits",
    "detector_oversampling", "detector_width", "courant", "wavelength",
    "slit_width", "slit_spacing", "source_angle_deg", "source_phase",
    "source_y_offset", "wavelength_nm", "slit_width_um", "slit_spacing_um",
    "screen_distance_m", "screen_half_height_mm", "um_to_cells", "cells_per_m",
    "lam_cells_min", "lam_cells_max", "slit_height_um", "slit_height", "coherent",
    "particle_mode",
)


def _build_library():
    compiler = shlex.split(os.environ.get("CXX", "g++"))
    if not compiler or shutil.which(compiler[0]) is None:
        raise RuntimeError(
            "The C++ backend needs a C++17 compiler (for example g++); "
            "install one or set the CXX environment variable."
        )
    command = compiler + ["-O3", "-DNDEBUG", "-std=c++17"]
    if sys.platform == "win32":
        command.append("-shared")
    elif sys.platform == "darwin":
        command.extend(["-dynamiclib", "-fPIC"])
    else:
        command.extend(["-shared", "-fPIC"])
    command.extend([
        os.path.join(_SOURCE_DIR, "physics.cpp"),
        os.path.join(_SOURCE_DIR, "physics_capi.cpp"),
        "-o",
        _LIBRARY_PATH,
    ])
    try:
        result = subprocess.run(
            command, cwd=_SOURCE_DIR, check=False, capture_output=True, text=True
        )
    except OSError as exc:
        raise RuntimeError(f"Could not start the C++ compiler: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"C++ backend build failed:\n{detail}")


def _load_library():
    global _LIBRARY, _DLL_DIRECTORY
    if _LIBRARY is not None:
        return _LIBRARY
    source_files = (
        os.path.join(_SOURCE_DIR, "physics.cpp"),
        os.path.join(_SOURCE_DIR, "physics_capi.cpp"),
        os.path.join(_SOURCE_DIR, "physics.h"),
    )
    newest_source = max(os.path.getmtime(path) for path in source_files)
    if not os.path.isfile(_LIBRARY_PATH) or os.path.getmtime(_LIBRARY_PATH) < newest_source:
        _build_library()
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY = os.add_dll_directory(_SOURCE_DIR)
    library = ctypes.CDLL(_LIBRARY_PATH)
    void_p, int_t, double_t = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    float_p = ctypes.POINTER(ctypes.c_float)
    byte_p = ctypes.POINTER(ctypes.c_uint8)
    library.sim_last_error.restype = ctypes.c_char_p
    library.sim_create.restype = void_p
    library.sim_destroy.argtypes = [void_p]
    library.sim_set_param.argtypes = [void_p, ctypes.c_char_p, double_t]
    library.sim_set_param.restype = int_t
    library.sim_build.argtypes = [void_p]
    library.sim_build.restype = int_t
    library.sim_reset.argtypes = [void_p]
    library.sim_reset.restype = int_t
    library.sim_step_n.argtypes = [void_p, int_t]
    library.sim_step_n.restype = int_t
    library.sim_t_step.argtypes = [void_p]
    library.sim_t_step.restype = ctypes.c_longlong
    library.sim_src_y.argtypes = [void_p]
    library.sim_src_y.restype = int_t
    library.sim_copy_field.argtypes = [void_p, float_p]
    library.sim_copy_field.restype = int_t
    library.sim_copy_detector.argtypes = [void_p, float_p]
    library.sim_copy_detector.restype = int_t
    library.sim_copy_slit_open.argtypes = [void_p, byte_p]
    library.sim_copy_slit_open.restype = int_t
    _LIBRARY = library
    return library


def _check(library, result):
    if result != 0:
        message = library.sim_last_error()
        detail = message.decode("utf-8", errors="replace") if message else "unknown C++ error"
        raise RuntimeError(f"C++ wave simulator failed: {detail}")


class CppWaveSim2D:
    label = "CPU C++"

    def __init__(self, p, device=None):
        self.p = p
        self.nx, self.ny = int(p.nx), int(p.ny)
        self._library = _load_library()
        self._handle = self._library.sim_create()
        if not self._handle:
            _check(self._library, -1)
        try:
            self._push()
            _check(self._library, self._library.sim_build(self._handle))
            self._field = np.empty((self.ny, self.nx), np.float32)
            self._det = np.empty(self.ny, np.float32)
            self._slit = np.empty(self.ny, np.uint8)
            self._refresh_static()
        except Exception:
            self._library.sim_destroy(self._handle)
            self._handle = None
            raise

    def _push(self):
        for name in _PARAM_NAMES:
            result = self._library.sim_set_param(
                self._handle, name.encode("ascii"), float(getattr(self.p, name))
            )
            _check(self._library, result)

    def _refresh_static(self):
        _check(
            self._library,
            self._library.sim_copy_slit_open(
                self._handle, self._slit.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            ),
        )
        self.slit_open = self._slit.astype(bool)
        self.src_y = np.array([self._library.sim_src_y(self._handle)])

    def reset(self):
        self._push()
        _check(self._library, self._library.sim_reset(self._handle))
        self._refresh_static()

    def step_n(self, n=1):
        _check(self._library, self._library.sim_step_n(self._handle, int(n)))

    def step(self):
        self.step_n(1)

    @property
    def t_step(self):
        return int(self._library.sim_t_step(self._handle))

    def field_slice(self):
        _check(
            self._library,
            self._library.sim_copy_field(
                self._handle, self._field.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ),
        )
        return self._field

    def detector_profile(self):
        _check(
            self._library,
            self._library.sim_copy_detector(
                self._handle, self._det.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ),
        )
        return self._det

    def detector_image(self):
        return self.detector_profile()

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self._library.sim_destroy(handle)
            self._handle = None
