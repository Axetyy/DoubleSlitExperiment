
import numpy as np

# u -  present state of field
# u_next - next state of field
# u_prev - prev state of field
# u->u_next->u_prev->u->u_next...


try:
    import torch
    HAVE_TORCH = True
except Exception:
    torch = None
    HAVE_TORCH = False
    
def cuda_available():
    return HAVE_TORCH and torch.cuda.is_available()

def pick_device(name=None):
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch is not installed (pip install torch).")
    if name in (None,"cuda","gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_label(device):
    if device.type == "cuda":
        return f"GPU torch ({torch.cuda.get_device_name(device)})"
    return "CPU torch"


class TorchWaveSim2D:
    AVG_STEPS = 480
    
    def __init__(self,p,device=None):        
        self.p = p
        self.device = pick_device(device)
        self.label = device_label(self.device)
        self.rng = np.random.default_rng()
        self.t_step = 0

        shape = (p.ny, p.nx)
        self.u = torch.zeros(shape,dtype=torch.float32,device=self.device)
        self.u_prev = torch.zeros_like(self.u)
        self.u_next = torch.zeros_like(self.u)
        self.I_det = torch.zeros(p.ny,dtype=torch.float32,device=self.device)

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
    
    def step_n(self,n=1):
        if n<=0:
            return  0
        p = self.p
        source_values = torch.from_numpy(self._source_values(n)).to(self.device)
        u,u_prev,u_next = self.u,self.u_prev,self.u_next
        c_next,c_prev = self.c_next,self.c_prev
        C2 = float(p.courant**2)
        sy, sx = int(self.src_y[0]), int(p.x_src)
        x0 = max(0, p.x_det - p.detector_width + 1)
        x1 = p.x_det
        inv_avg = 1.0 / self.AVG_STEPS
        
        with torch.no_grad():
            for k in range(n):
                # un = cn * (2u + C2 * lap(u)) - cp * up      (lap = 0 on the border)
                torch.mul(u,2.0,out=u_next)
                lap = u[2:, 1:-1] + u[:-2, 1:-1]
                lap += u[1:-1, 2:]
                lap += u[1:-1, :-2]
                lap.sub_(u[1:-1, 1:-1], alpha=4.0)
                u_next[1:-1, 1:-1].add_(lap, alpha=C2)
                u_next.mul_(c_next)
                u_next.addcmul_(c_prev,u_prev,value=-1.0)
                u_next[sy,sx].add_(source_values[k])
                
                instance = u_next[:,x0:x1+1].pow(2).mean(dim=1)
                self.I_det += (instance - self.I_det) * inv_avg
                
                u, u_prev, u_next = u_next, u, u_prev
        self.u,self.u_prev,self.u_next = u,u_prev,u_next
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
    