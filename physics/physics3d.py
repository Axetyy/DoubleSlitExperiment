import numpy as np
import os

# u -  present state of field
# u_next - next state of field
# u_prev - prev state of field
# u->u_next->u_prev->u->u_next...

# It's just physics.py from 2d  + numba parallelization + another 3d term for the equation
# 4-> 8 neighbours in space
# i+-1,j-+1,k+-1 u[i][j][k]
# d2u/dt2 = c^2 (d2u/dx2 + d2u/dy2 + d2u/dz2) - sigma du/dt + source
# the detector is the plane x = x_det; its intensity is an (ny, nz) image.

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
PARALLEL = os.environ.get("WAVESIM_PARALLEL", "1") != "0"
 
import numpy as np
 
try:
    from numba import njit, prange
 
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
 
try:
    import torch
 
    HAVE_TORCH = True
except Exception:
    torch = None
    HAVE_TORCH = False
    
### NUMBA KERNEL (CPU)    

if HAVE_NUMBA:
    @njit(parallel=PARALLEL, cache=False)
    def _one_step_3d(u, up, un, c_next, c_prev, C2, sz, sy, sx, sv, x0, x1, I_det, avg):
        
        nz,ny, nx = u.shape
        two = np.float32(2.0)
        six = np.float32(6.0)
        inv_w = 1.0 / (x1 - x0 + 1)

        for k in prange(nz):
            for i in range(ny):
                if k == 0 or k == nz-1 or i == 0 or i == ny-1:
                    for j in range(nx):
                        un[k, i, j] = c_next[k, i, j] * (two * u[k,i, j]) - c_prev[k,i, j] * up[k,i, j]
                else:
                    un[k,i, 0] = c_next[k,i, 0] * (two * u[k,i, 0]) - c_prev[k,i, 0] * up[k,i, 0]
                    un[k,i, nx - 1] = (
                        c_next[k,i, nx - 1] * (two * u[k,i, nx - 1])
                        - c_prev[k,i, nx - 1] * up[k,i, nx - 1]
                    )
                    for j in range(1, nx - 1):
                        lap = C2 * (
                            u[k + 1, i, j] + u[k - 1, i, j]
                            + u[k, i + 1, j] + u[k, i - 1, j]
                            + u[k, i, j + 1] + u[k, i, j - 1]
                            - six * u[k, i, j]
                        )
                        un[k,i, j] = c_next[k,i, j] * (two * u[k,i, j] + lap) - c_prev[k,i, j] * up[k,i, j]

                if k == sz and i == sy:
                    un[k, i, sx] += sv  
                    
                s = 0.0
                for j in range(x0, x1 + 1):
                    v = un[k,i, j]
                    s += v * v
                I_det[i,k] += np.float32((s * inv_w - I_det[i,k]) / avg)


    @njit(cache=False)
    def _run_steps_3d(u, up, un, cn, cp, C2, sz, sy, sx, sv_arr, x0, x1, I_det, avg):
        for n in range(sv_arr.shape[0]):
            _one_step_3d(u, up, un, cn, cp, C2, sz, sy, sx, sv_arr[n], x0, x1, I_det, avg)
            u, up, un = un, u, up
        return u, up, un

    _WARMED = [False]
    def _warmup():
        if _WARMED[0]:
            return
        a = np.zeros((6, 6, 8), np.float32)
        b = np.zeros_like(a)
        c = np.zeros_like(a)
        ones = np.ones_like(a)
        I = np.zeros((6, 6), np.float32)
        _run_steps_3d(a, b, c, ones, ones, np.float32(0.25), 3, 3, 2,
                      np.zeros(1, np.float32), 4, 6, I, 480.0)
        _WARMED[0] = True


# Shared base

class _Base3d:    
    AVG_STEPS = 480
    label = ""
    def __init__(self,p):
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
        edge = np.maximum(np.maximum(dz, dy), dx)
        damping = (0.6 * edge ** 2).astype(np.float32)
        self.dprev = (1 - 0.5 * damping).astype(np.float32)
        self.dnext = (1 / (1 + 0.5 * damping)).astype(np.float32)
        
    def rebuild(self, fresh=False):
        p = self.p
        yc,zc = p.ny / 2, p.nz / 2
        y = np.arange(p.ny)
        z = np.arange(p.nz)

        self.slit_centers = (
            yc + (np.arange(p.n_slits) - (p.n_slits - 1) / 2) * p.slit_spacing
        )
        open_y = np.zeros(p.ny,dtype=bool)
        for c in self.slit_centers:
            open_y |= np.abs(y - c) <= p.slit_width / 2
        open_z = np.abs(z-zc) <= p.slit_height / 2
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
        mask[:, :, p.x_wall:p.x_wall + p.wall_thickness] = (
            aperture
        )[:, :, None]

        tan_t = np.tan(np.deg2rad(p.source_angle_deg))
        yc_src = np.clip(
            yc - (p.x_wall - p.x_src) * tan_t,
            p.sponge,
            p.ny - p.sponge - 1,
        )
        
        self.src_y = np.array([int(round(yc_src))])
        self.src_z = int(p.nz // 2)
        k = 2 * np.pi / p.wavelength

        self.phase_offset = (
            -k * (self.src_y - yc_src) * np.sin(np.deg2rad(p.source_angle_deg))
        )
        self.phase_noise = 0.0
        self.omega = 2 * np.pi / p.wavelength
        self.src_mask = float(mask[self.src_z,self.src_y[0],p.x_src])
        
        cn = self.dnext * mask
        cp = self.dprev * self.dnext * mask
        self._upload_coeffs(cn, cp)
        del mask, cn, cp
 
        self.settle = int((p.x_det - p.x_src) / p.courant + 2 * self.AVG_STEPS)
        if fresh:
            self.settle += int((p.x_wall - p.x_src) / p.courant)
    def reset(self):
        self._zero_state()
        self.t_step = 0
        self.rebuild(True)
    def _source_values(self,n):
        p = self.p
        t = self.t_step + np.arange(n)
        phase = self.omega * p.courant * t + self.phase_offset
        if not p.coherent:
            noise = self.phase_noise + np.cumsum(0.1 * self.rng.standard_normal(n))
            self.phase_noise = float(noise[-1])
            phase = phase + noise
        amplitude = np.minimum(1.0, t / 100)
        return (amplitude * np.sin(phase) * self.src_mask).astype(np.float32)
    def step_n(self,n=1):
        if n <=0:
            return 0
        self._advance(self._source_values(n))
        self.t_step +=n
        self.settle -=n
    def step(self):
        self.step_n(self,1)



class WaveSim3D(_Base3d):
    label = "CPU numba"
    
    def __init__(self,p,device=None):       
        if not HAVE_NUMBA:
             raise RuntimeError("numba is not installed (pip install numba).")
        _warmup()
        super().__init__(p)
    def _alloc_state(self):
        shape = (self.nz,self.ny,self.nx)
        self.u = np.zeros(shape,np.float32)
        self.u_prev = np.zeros(shape,np.float32)
        self.u_next = np.zeros(shape,np.float32)
        self.I_det = np.zeros((self.ny,self.nz),np.float32)
        
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
        dev = self.device
        self.u = torch.zeros(shape, dtype=torch.float32, device=dev)
        self.u_prev = torch.zeros_like(self.u)
        self.u_next = torch.zeros_like(self.u)
        self.I_det = torch.zeros((self.ny, self.nz), dtype=torch.float32, device=dev)
 
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
        cn, cp = self.cn, self.cp
        C2 = float(p.courant ** 2)
        sz, sy, sx = int(self.src_z), int(self.src_y[0]), int(p.x_src)
        x0 = max(0, p.x_det - p.detector_width + 1)
        x1 = p.x_det
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
                un.mul_(cn)
                un.addcmul_(cp, up, value=-1.0)
                un[sz, sy, sx].add_(sv[n])
 
                inst = un[:, :, x0:x1 + 1].pow(2).mean(dim=2)
                self.I_det += (inst.T - self.I_det) * inv_avg
 
                u, up, un = un, u, up
 
        self.u, self.u_prev, self.u_next = u, up, un
 
    def field_slice(self):
        return self.u[self.nz // 2].detach().cpu().numpy()
 
    def detector_image(self):
        return self.I_det.detach().cpu().numpy()
 
    def detector_profile(self):
        return self.I_det[:, self.nz // 2].detach().cpu().numpy()