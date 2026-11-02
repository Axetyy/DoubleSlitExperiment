"""Simulation state and backend orchestration.

The controller deliberately does not know anything about Matplotlib.  A UI can
consume :meth:`snapshot` and render the returned arrays in any format.
"""

import numpy as np

from config import Params, params_3d
from sampling.sampling import ParticleSampler
from sampling.sampling3d import ParticleSampler3D
from utils.reference import fraunhofer, fraunhofer3d
from utils.utils import wavelength_to_rgb

try:
    from physics.core import TorchWaveSim2D, cuda_available
except Exception:
    TorchWaveSim2D = None

    def cuda_available():
        return False

try:
    from physics.core import WaveSimulator as NumbaSim2D
except Exception:
    NumbaSim2D = None

try:
    from physics.core import CppWaveSim2D
except Exception:
    CppWaveSim2D = None

try:
    import physics.core as physics3d
except Exception:
    physics3d = None


class SimulationManager:
    """Own simulation parameters, backend objects, and simulation state."""

    def __init__(self, dim=2, device="CPU"):
        self.dim = dim
        self.device = device
        self.p2 = Params()
        self.p3 = params_3d()
        self.p = self.p2 if dim == 2 else self.p3
        self.sim = None
        self.sampler = None
        self.backend_label = ""
        self.screen_rows = np.arange(self.p.ny)
        self.screen_y_mm = np.zeros(self.p.ny)
        self.screen_z_half = 1.0
        self.screen_pixel_size_mm = 1.0
        self.screen_arrival_step = 0
        self.screen_color = wavelength_to_rgb(self.p.wavelength_nm)

    @property
    def cuda_available(self):
        return cuda_available()

    @property
    def particle_mode(self):
        return bool(self.p.particle_mode)

    def build(self):
        """Create the selected numerical backend and its particle sampler."""
        if self.device == "GPU" and not cuda_available():
            raise RuntimeError("CUDA GPU is not available.")

        if self.dim == 2:
            if self.device == "GPU":
                if TorchWaveSim2D is None:
                    raise RuntimeError("PyTorch backend unavailable.")
                simulation = TorchWaveSim2D(self.p, "cuda")
            elif self.device == "C++":
                if CppWaveSim2D is None:
                    raise RuntimeError("C++ backend unavailable.")
                simulation = CppWaveSim2D(self.p)
            elif NumbaSim2D is not None:
                simulation = NumbaSim2D(self.p)
                simulation.label = "CPU numba"
            elif TorchWaveSim2D is not None:
                simulation = TorchWaveSim2D(self.p, "cpu")
            else:
                raise RuntimeError("Install numba or torch.")
            sampler = ParticleSampler(self.p.ny, self.p.x_det)
        else:
            if physics3d is None:
                raise RuntimeError("3D physics backend unavailable.")
            if self.device == "GPU":
                simulation = physics3d.TorchWaveSim3D(self.p, "cuda")
            elif physics3d.HAVE_NUMBA:
                simulation = physics3d.WaveSim3D(self.p)
            else:
                simulation = physics3d.TorchWaveSim3D(self.p, "cpu")
            sampler = ParticleSampler3D(self.p.ny, self.p.nz, self.p.x_det)

        self.sim = simulation
        self.sampler = sampler
        self.backend_label = f"{self.dim}D | {getattr(simulation, 'label', self.device)}"
        self.configure_geometry()

    def configure_geometry(self):
        """Recalculate detector coordinates after a parameter or dimension change."""
        view_ny = min(self.p.ny, self.p.view_ny) if self.dim == 2 else self.p.ny
        view_start = (self.p.ny - view_ny) // 2
        self.screen_rows = np.arange(view_start, view_start + view_ny)
        half = self.p.screen_half_height_mm
        self.screen_pixel_size_mm = 2 * half / view_ny
        self.screen_y_mm = (
            (np.arange(view_ny) + 0.5) * self.screen_pixel_size_mm - half
        )
        self.screen_z_half = self.p.nz * self.screen_pixel_size_mm / 2
        self.screen_color = wavelength_to_rgb(self.p.wavelength_nm)
        self.screen_arrival_step = int(
            np.ceil((self.p.x_det - self.p.x_src) / self.p.courant)
        )
        if self.sampler is not None:
            self.sampler.x_detector = self.p.x_det

    def select(self, dim=None, device=None, coherent=None, particle_mode=None):
        """Apply UI selections and rebuild the simulation if needed."""
        if dim is not None:
            self.dim = dim
            self.p = self.p3 if dim == 3 else self.p2
        if device is not None:
            self.device = device
        if coherent is not None:
            self.p.coherent = coherent
        if particle_mode is not None:
            self.p.particle_mode = particle_mode
        self.build()

    def reset(self):
        if self.sim is None:
            return
        self.sim.reset()
        self.sampler.reset()

    def step(self, steps, photons_per_frame=0):
        """Advance the wave and optionally sample arriving particles."""
        if self.sim is None:
            raise RuntimeError("The simulation has not been built.")
        self.sim.step_n(int(steps))
        if (
            self.particle_mode
            and photons_per_frame
            and self.sim.t_step >= self.screen_arrival_step
        ):
            profile = self.detector_image() if self.dim == 3 else self.detector_profile()
            self.sampler.sample(profile, int(photons_per_frame))

    def field(self):
        return self.sim.field_slice() if hasattr(self.sim, "field_slice") else self.sim.u

    def detector_profile(self):
        return (
            self.sim.detector_profile()
            if hasattr(self.sim, "detector_profile")
            else self.sim.I_det
        )

    def detector_image(self):
        return self.sim.detector_image()

    def reference_profile(self):
        if self.dim == 2:
            return fraunhofer(self.p, self.screen_rows, self.sim.slit_open, self.sim.src_y[0])
        return fraunhofer3d(
            self.p,
            self.screen_rows,
            self.sim.slit_open,
            self.sim.slit_open_z,
            self.sim.src_y[0],
            self.sim.src_z,
        )

    def snapshot(self):
        """Return all data needed by a renderer without exposing backend details."""
        return {
            "dimension": self.dim,
            "field": np.asarray(self.field()),
            "detector_profile": np.asarray(self.detector_profile()),
            "detector_image": (
                np.asarray(self.detector_image()) if self.dim == 3 else None
            ),
            "reference_profile": np.asarray(self.reference_profile()),
            "screen_rows": self.screen_rows.copy(),
            "screen_y_mm": self.screen_y_mm.copy(),
            "screen_color": self.screen_color,
            "particle_mode": self.particle_mode,
            "particle_hist": self.sampler.hist.copy(),
            "recent_hits": self.sampler.recent.copy(),
            "time_step": self.sim.t_step,
            "arrival_step": self.screen_arrival_step,
        }
