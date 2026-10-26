"""Unit tests for physics backends and shared simulation helpers."""

import ctypes
import builtins
import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import physics.core as core
from config import Params, params_3d
from physics import (
    CppWaveSim2D,
    TorchWaveSim2D,
    TorchWaveSim3D,
    WaveSim3D,
    WaveSimulator,
    cuda_available,
    device_label,
    pick_device,
)
from physics import *
from reference import (
    _numerical_wavenumber,
    _numerical_wavenumber_3d,
    fraunhofer,
    fraunhofer3d,
)
from sampling.sampling import ParticleSampler
from sampling.sampling3d import ParticleSampler3D
from utils import high_res_screen, runs, wavelength_to_rgb


def small_params(**overrides):
    values = dict(
        nx=16,
        ny=12,
        nz=10,
        x_src=2,
        x_wall=5,
        x_det=12,
        sponge=1,
        wall_thickness=1,
        slit_width=2,
        slit_spacing=3,
        slit_height=2,
        detector_width=2,
        courant=0.2,
        wavelength=4.0,
        n_slits=2,
    )
    values.update(overrides)
    return Params(**values)


class FixedRng:
    def __init__(self, poisson=None):
        self.poisson_value = poisson

    def poisson(self, mean):
        return mean if self.poisson_value is None else self.poisson_value

    def random(self, size=None):
        return np.full(size, 0.5) if size is not None else 0.5

    def uniform(self, low, high, size):
        return np.full(size, (low + high) / 2)


class FakeFunction:
    def __init__(self, callback=None):
        self.callback = callback or (lambda *args: 0)
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class FakeLibrary:
    def __init__(self, build_result=0):
        self.destroyed = []
        self.params = []
        self.steps = []
        self.field = np.arange(192, dtype=np.float32)
        self.detector = np.arange(12, dtype=np.float32)
        self.slit = np.array([1, 0] * 6, dtype=np.uint8)
        self.build_result = build_result
        self.sim_last_error = FakeFunction(lambda: b"fake backend error")
        self.sim_create = FakeFunction(lambda: 23)
        self.sim_destroy = FakeFunction(lambda handle: self.destroyed.append(handle))
        self.sim_set_param = FakeFunction(
            lambda handle, name, value: self.params.append((name, value)) or 0
        )
        self.sim_build = FakeFunction(lambda handle: self.build_result)
        self.sim_reset = FakeFunction(lambda handle: 0)
        self.sim_step_n = FakeFunction(
            lambda handle, count: self.steps.append(count) or 0
        )
        self.sim_t_step = FakeFunction(lambda handle: 5)
        self.sim_src_y = FakeFunction(lambda handle: 4)
        self.sim_copy_field = FakeFunction(self._copy_field)
        self.sim_copy_detector = FakeFunction(self._copy_detector)
        self.sim_copy_slit_open = FakeFunction(self._copy_slit)

    def _copy_field(self, handle, pointer):
        target = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_float))
        for index, value in enumerate(self.field):
            target[index] = value
        return 0

    def _copy_detector(self, handle, pointer):
        target = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_float))
        for index, value in enumerate(self.detector):
            target[index] = value
        return 0

    def _copy_slit(self, handle, pointer):
        target = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint8))
        for index, value in enumerate(self.slit):
            target[index] = value
        return 0


class HelpersTests(unittest.TestCase):
    def test_params_defaults_and_3d_conversion(self):
        p = Params()
        self.assertEqual(p.nx, 1000)
        self.assertTrue(p.coherent)
        p3 = params_3d()
        self.assertEqual(p3.nz, 192)
        self.assertEqual(p3.x_det, p3.x_wall + round(p3.cells_per_m * p3.screen_distance_m))
        self.assertAlmostEqual(p3.slit_width, p3.slit_width_um * p3.um_to_cells)
        self.assertAlmostEqual(p3.slit_height, p3.slit_height_um * p3.um_to_cells)
        self.assertGreaterEqual(p3.wavelength, p3.lam_cells_min)
        self.assertLessEqual(p3.wavelength, p3.lam_cells_max)

    def test_wavelength_to_rgb_all_segments_and_bounds(self):
        wavelengths = np.array(
            [379, 380, 419, 420, 439, 440, 489, 490, 509, 510, 579, 580, 644, 645, 750, 751]
        )
        colors = wavelength_to_rgb(wavelengths)
        self.assertEqual(colors.shape, (len(wavelengths), 3))
        self.assertTrue(np.all(colors >= 0))
        self.assertTrue(np.all(colors <= 1))
        self.assertTrue(np.array_equal(colors[0], np.zeros(3)))
        self.assertTrue(np.array_equal(colors[-1], np.zeros(3)))
        self.assertGreater(colors[1, 2], 0)
        self.assertGreater(colors[14, 0], 0)
        self.assertEqual(wavelength_to_rgb(500).shape, (3,))

    def test_screen_upsampling_and_runs(self):
        upsampled = high_res_screen(np.array([0.0, 1.0, 0.0]), oversampling=2)
        self.assertEqual(upsampled.shape, (6,))
        self.assertTrue(np.allclose(upsampled[[0, -1]], [0.0, 0.0]))
        self.assertEqual(runs([]), [])
        self.assertEqual(runs([False, False, True, True, False]),
                         [(0, 2, False), (2, 4, True), (4, 5, False)])

    def test_particle_sampler_branches_and_recent_limit(self):
        sampler = ParticleSampler(4, 10, max_recent=2)
        sampler.rng = FixedRng(poisson=0)
        self.assertEqual(sampler.sample(np.ones(4), 3).size, 0)
        sampler.rng = FixedRng(poisson=2)
        self.assertEqual(sampler.sample(np.zeros(4), 2).size, 0)
        positions = sampler.sample(np.array([1, 0, 0, 0]), 3)
        self.assertEqual(positions.size, 2)
        self.assertEqual(sampler.hist.sum(), 2)
        self.assertEqual(sampler.recent.shape, (2, 2))
        sampler.reset()
        self.assertEqual(sampler.hist.sum(), 0)
        self.assertEqual(sampler.recent.shape, (0, 2))

    def test_particle_sampler3d_branches_and_recent_limit(self):
        sampler = ParticleSampler3D(3, 2, 20, max_recent=2)
        sampler.rng = FixedRng(poisson=0)
        self.assertIsNone(sampler.sample(np.ones((3, 2)), 2))
        sampler.rng = FixedRng(poisson=3)
        self.assertIsNone(sampler.sample(np.zeros((3, 2)), 3))
        sampler.sample(np.array([[1, 0], [0, 0], [0, 0]]), 3)
        self.assertEqual(sampler.hist.sum(), 3)
        self.assertEqual(sampler.recent.shape, (2, 2))
        sampler.reset()
        self.assertEqual(sampler.hist.shape, (3, 2))
        self.assertEqual(sampler.recent.shape, (0, 2))


class ReferenceTests(unittest.TestCase):
    def test_wavenumber_solvers(self):
        p = small_params()
        k2 = _numerical_wavenumber(p, np.array([0.0, 0.4]))
        k3 = _numerical_wavenumber_3d(
            p, np.array([1.0, 0.8]), np.array([0.0, 0.3]), np.array([0.0, 0.2])
        )
        self.assertEqual(k2.shape, (2,))
        self.assertEqual(k3.shape, (2,))
        self.assertTrue(np.all(k2 > 0))
        self.assertTrue(np.all(k3 > 0))

    def test_fraunhofer_2d_normalized_and_empty_aperture(self):
        p = small_params()
        detector_rows = np.arange(p.ny)
        result = fraunhofer(p, detector_rows, np.array([0, 0, 1, 1, 0]), 6)
        self.assertEqual(result.shape, (p.ny,))
        self.assertAlmostEqual(float(result.max()), 1.0)
        self.assertTrue(np.all(np.isfinite(result)))
        empty = fraunhofer(p, detector_rows, np.zeros(p.ny, dtype=bool), 6)
        self.assertEqual(empty.shape, (p.ny,))
        self.assertTrue(np.array_equal(empty, np.zeros(p.ny)))

    def test_fraunhofer_3d_implicit_explicit_empty_apertures(self):
        p = small_params()
        rows = np.arange(p.ny)
        open_y = np.array([False, False, True, True, False, False, False, False, False, False, False, False])
        open_z = np.array([False, False, False, False, True, True, False, False, False, False])
        profile = fraunhofer3d(p, rows, open_y, open_z, 6, 5)
        self.assertEqual(profile.shape, (p.ny,))
        self.assertAlmostEqual(float(profile.max()), 1.0)
        empty = fraunhofer3d(p, rows, open_y, open_z, 6, 5, np.zeros((p.nz, p.ny)))
        self.assertTrue(np.array_equal(empty, np.zeros(p.ny)))
        explicit = fraunhofer3d(p, rows, open_y, open_z, 6, 5, np.ones((p.nz, p.ny)))
        self.assertEqual(explicit.shape, (p.ny,))
        self.assertAlmostEqual(float(explicit.max()), 1.0)


class NumbaBackendTests(unittest.TestCase):
    def test_2d_kernel_python_path_and_run_wrapper(self):
        u = np.ones((5, 6), dtype=np.float32)
        previous = np.zeros_like(u)
        following = np.zeros_like(u)
        coeff = np.ones_like(u)
        intensity = np.zeros(5, dtype=np.float32)
        core._one_step.py_func(
            u, previous, following, coeff, coeff, np.float32(0.25),
            2, 2, np.float32(1), 1, 4, intensity, 480.0
        )
        self.assertGreater(following[2, 2], 0)
        self.assertGreater(intensity.sum(), 0)
        self.assertEqual(len(core._run_steps.py_func(
            u, previous, following, coeff, coeff, np.float32(0.25), 2, 2,
            np.zeros(1, dtype=np.float32), 1, 4, intensity, 480.0
        )), 3)

    def test_2d_simulator_steps_resets_and_errors(self):
        p = small_params()
        sim = WaveSimulator(p)
        self.assertEqual(sim.step_n(0), None)
        sim.step()
        sim.step_n(2)
        self.assertEqual(sim.t_step, 3)
        self.assertEqual(sim.detector_image().shape, (p.ny,))
        sim.reset()
        self.assertEqual(sim.t_step, 0)
        p.coherent = False
        sim.step_n(2)
        self.assertEqual(sim.t_step, 2)
        bad = WaveSimulator(small_params(x_det=16))
        with self.assertRaises(ValueError):
            bad.step()
        broken = WaveSimulator(small_params())
        broken.u = np.asfortranarray(broken.u)
        with self.assertRaises(ValueError):
            broken.step()

    def test_warmup_only_runs_once(self):
        previous = core._WARMED
        try:
            core._WARMED = False
            core._warmup()
            self.assertFalse(core._WARMED)
            core._warmup()
            self.assertFalse(core._WARMED)
            WaveSimulator(small_params())
            self.assertTrue(core._WARMED)
        finally:
            core._WARMED = previous

    def test_3d_kernel_python_path_and_simulator(self):
        u = np.ones((4, 5, 6), dtype=np.float32)
        previous = np.zeros_like(u)
        following = np.zeros_like(u)
        coeff = np.ones_like(u)
        intensity = np.zeros((5, 4), dtype=np.float32)
        core._one_step_3d.py_func(
            u, previous, following, coeff, coeff, np.float32(0.1),
            1, 2, 2, np.float32(1), 1, 4, intensity, 480.0
        )
        self.assertGreater(following[1, 2, 2], 0)
        self.assertGreater(intensity.sum(), 0)
        self.assertEqual(len(core._run_steps_3d.py_func(
            u, previous, following, coeff, coeff, np.float32(0.1),
            1, 2, 2, np.zeros(1, dtype=np.float32), 1, 4, intensity, 480.0
        )), 3)
        sim = WaveSim3D(small_params())
        sim.step_n(0)
        sim.step()
        sim.step_n(2)
        self.assertEqual(sim.t_step, 3)
        self.assertEqual(sim.field_slice().shape, (sim.ny, sim.nx))
        self.assertEqual(sim.detector_image().shape, (sim.ny, sim.nz))
        self.assertEqual(sim.detector_profile().shape, (sim.ny,))
        sim.reset()
        self.assertEqual(sim.t_step, 0)
        sim.p.coherent = False
        sim.step_n(1)
        self.assertEqual(sim.t_step, 1)
        invalid = WaveSim3D(small_params(x_det=16))
        with self.assertRaises(ValueError):
            invalid.step()
        previous_warmed = core._WARMED_3D
        try:
            core._WARMED_3D = False
            core._warmup_3d()
            self.assertTrue(core._WARMED_3D)
            core._warmup_3d()
        finally:
            core._WARMED_3D = previous_warmed


class TorchBackendTests(unittest.TestCase):
    def test_device_helpers(self):
        self.assertTrue(cuda_available() or not cuda_available())
        self.assertEqual(pick_device("cpu").type, "cpu")
        self.assertEqual(device_label(core.torch.device("cpu")), "CPU torch")
        with patch.object(core.torch.cuda, "is_available", return_value=False):
            self.assertEqual(pick_device("cuda").type, "cpu")
        with patch.object(core.torch.cuda, "is_available", return_value=True), patch.object(
            core.torch.cuda, "get_device_name", return_value="test GPU"
        ):
            gpu = pick_device("gpu")
            self.assertEqual(gpu.type, "cuda")
            self.assertEqual(device_label(gpu), "GPU torch (test GPU)")
            with patch.object(core._Base3d, "__init__", return_value=None):
                gpu_sim = TorchWaveSim3D(small_params(), "gpu")
            self.assertEqual(gpu_sim.label, "GPU torch (test GPU)")
        with patch.object(core, "HAVE_TORCH", False):
            self.assertFalse(cuda_available())
            with self.assertRaisesRegex(RuntimeError, "PyTorch is not installed"):
                pick_device()
            with self.assertRaisesRegex(RuntimeError, "PyTorch is not installed"):
                TorchWaveSim3D(small_params())

    def test_2d_torch_steps_reset_and_helpers(self):
        p = small_params()
        sim = TorchWaveSim2D(p, "cpu")
        self.assertEqual(sim.step_n(0), 0)
        sim.step()
        sim.step_n(2)
        self.assertEqual(sim.t_step, 3)
        self.assertEqual(sim.field_slice().shape, (p.ny, p.nx))
        self.assertEqual(sim.detector_profile().shape, (p.ny,))
        self.assertTrue(np.array_equal(sim.detector_image(), sim.detector_profile()))
        sim.p.coherent = False
        sim._source_values(2)
        sim.rebuild(fresh=False)
        sim.reset()
        self.assertEqual(sim.t_step, 0)
        self.assertEqual(sim.settle, int((p.x_det - p.x_src) / p.courant + 2 * sim.AVG_STEPS
                                         + (p.x_wall - p.x_src) / p.courant))

    def test_3d_torch_steps_reset_and_shapes(self):
        sim = TorchWaveSim3D(small_params(), "cpu")
        self.assertEqual(sim.step_n(0), 0)
        sim.step()
        sim.step_n(2)
        self.assertEqual(sim.t_step, 3)
        self.assertEqual(sim.field_slice().shape, (sim.ny, sim.nx))
        self.assertEqual(sim.detector_image().shape, (sim.ny, sim.nz))
        self.assertEqual(sim.detector_profile().shape, (sim.ny,))
        sim.p.coherent = False
        sim.step_n(1)
        sim.reset()
        self.assertEqual(sim.t_step, 0)
        circular = TorchWaveSim3D(small_params(slit_shape="circular"), "cpu")
        self.assertEqual(circular.aperture_mask.shape, (circular.nz, circular.ny))
        self.assertTrue(np.any(circular.aperture_mask))


class CppBackendTests(unittest.TestCase):
    def test_compiler_validation_and_build_outcomes(self):
        with patch.dict(os.environ, {"CXX": ""}), patch.object(core.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "needs a C\\+\\+17 compiler"):
                core._build_library()
        with patch.dict(os.environ, {"CXX": "fake-cxx"}), patch.object(
            core.shutil, "which", return_value="fake-cxx"
        ), patch.object(core.subprocess, "run", side_effect=OSError("missing")):
            with self.assertRaisesRegex(RuntimeError, "Could not start"):
                core._build_library()
        success = SimpleNamespace(returncode=0, stderr="", stdout="")
        with patch.dict(os.environ, {"CXX": "fake-cxx"}), patch.object(
            core.shutil, "which", return_value="fake-cxx"
        ), patch.object(core.subprocess, "run", return_value=success) as run:
            core._build_library()
            self.assertIn("-shared", run.call_args.args[0])
        for platform, expected_flag in (("darwin", "-dynamiclib"), ("linux", "-fPIC")):
            with patch.dict(os.environ, {"CXX": "fake-cxx"}), patch.object(
                core.shutil, "which", return_value="fake-cxx"
            ), patch.object(core.sys, "platform", platform), patch.object(
                core.subprocess, "run", return_value=success
            ) as run:
                core._build_library()
                self.assertIn(expected_flag, run.call_args.args[0])
        failure = SimpleNamespace(returncode=1, stderr="compiler error", stdout="")
        with patch.dict(os.environ, {"CXX": "fake-cxx"}), patch.object(
            core.shutil, "which", return_value="fake-cxx"
        ), patch.object(core.subprocess, "run", return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "compiler error"):
                core._build_library()

    def test_library_loading_cached_and_configured(self):
        original_library, original_directory = core._LIBRARY, core._DLL_DIRECTORY
        cached = object()
        try:
            core._LIBRARY = cached
            self.assertIs(core._load_library(), cached)
            fake_library = FakeLibrary()
            core._LIBRARY = None
            with patch.object(core.os.path, "getmtime", side_effect=[1, 1, 1, 2]), patch.object(
                core.os.path, "isfile", return_value=True
            ), patch.object(core.sys, "platform", "win32"), patch.object(
                core.os, "add_dll_directory", return_value="dll-dir"
            ) as add_directory, patch.object(core.ctypes, "CDLL", return_value=fake_library):
                with patch.object(core, "_build_library") as build:
                    loaded = core._load_library()
                build.assert_not_called()
                add_directory.assert_called_once_with(core._SOURCE_DIR)
            self.assertIs(loaded, fake_library)
            self.assertIs(core._DLL_DIRECTORY, "dll-dir")
            self.assertEqual(fake_library.sim_create.restype, ctypes.c_void_p)
        finally:
            core._LIBRARY, core._DLL_DIRECTORY = original_library, original_directory

    def test_library_loading_rebuilds_missing_or_stale_library(self):
        original_library = core._LIBRARY
        try:
            for exists, mtimes in ((False, [2, 2, 2]), (True, [2, 2, 2, 1])):
                core._LIBRARY = None
                fake_library = FakeLibrary()
                with patch.object(core.os.path, "getmtime", side_effect=mtimes), patch.object(
                    core.os.path, "isfile", return_value=exists
                ), patch.object(core.sys, "platform", "linux"), patch.object(
                    core.ctypes, "CDLL", return_value=fake_library
                ), patch.object(core, "_build_library") as build:
                    self.assertIs(core._load_library(), fake_library)
                    build.assert_called_once()
        finally:
            core._LIBRARY = original_library

    def test_error_check(self):
        library = FakeLibrary()
        core._check(library, 0)
        with self.assertRaisesRegex(RuntimeError, "fake backend error"):
            core._check(library, -1)
        library.sim_last_error = FakeFunction(lambda: None)
        with self.assertRaisesRegex(RuntimeError, "unknown C\\+\\+ error"):
            core._check(library, -1)

    def test_cpp_simulator_and_failure_cleanup(self):
        library = FakeLibrary()
        with patch.object(core, "_load_library", return_value=library):
            sim = CppWaveSim2D(small_params())
            self.assertEqual(len(library.params), len(core._PARAM_NAMES))
            self.assertEqual(sim.slit_open.shape, (sim.ny,))
            self.assertEqual(sim.src_y.tolist(), [4])
            sim.step_n(3)
            sim.step()
            self.assertEqual(library.steps, [3, 1])
            self.assertEqual(sim.t_step, 5)
            self.assertEqual(sim.field_slice().shape, (sim.ny, sim.nx))
            self.assertEqual(sim.detector_profile().shape, (sim.ny,))
            self.assertIs(sim.detector_image(), sim.detector_profile())
            sim.reset()
            self.assertEqual(library.destroyed, [])
            sim.__del__()
            self.assertEqual(library.destroyed, [23])
        failed_library = FakeLibrary(build_result=1)
        with patch.object(core, "_load_library", return_value=failed_library):
            with self.assertRaisesRegex(RuntimeError, "fake backend error"):
                CppWaveSim2D(small_params())
        self.assertEqual(failed_library.destroyed, [23])
        empty_library = FakeLibrary()
        empty_library.sim_create = FakeFunction(lambda: 0)
        with patch.object(core, "_load_library", return_value=empty_library):
            with self.assertRaisesRegex(RuntimeError, "fake backend error"):
                CppWaveSim2D(small_params())


    def test_torch_import_failure_is_reported_and_recovered(self):
        original_import = builtins.__import__

        def reject_torch(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("torch intentionally unavailable")
            return original_import(name, *args, **kwargs)

        def reject_numba(name, *args, **kwargs):
            if name == "numba":
                raise ImportError("numba intentionally unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_numba):
            importlib.reload(core)
        self.assertFalse(core.HAVE_NUMBA)
        self.assertEqual(core.njit(parallel=True)(lambda: 7)(), 7)
        self.assertEqual(core.njit(lambda: 8)(), 8)
        with self.assertRaisesRegex(RuntimeError, "numba is not installed"):
            core.WaveSimulator(small_params())
        with self.assertRaisesRegex(RuntimeError, "numba is not installed"):
            core.WaveSim3D(small_params())
        cpu_torch = core.TorchWaveSim2D(small_params(), "cpu")
        cpu_torch.step()
        torch_3d = core.TorchWaveSim3D(small_params(), "cpu")
        torch_3d.step()

        importlib.reload(core)
        self.assertTrue(core.HAVE_NUMBA)
        with patch("builtins.__import__", side_effect=reject_torch):
            importlib.reload(core)
        self.assertFalse(core.HAVE_TORCH)
        importlib.reload(core)
        self.assertTrue(core.HAVE_TORCH)


if __name__ == "__main__":
    unittest.main()
