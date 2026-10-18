"""ctypes wrapper for the optional C++ 2D wave simulator."""

import ctypes
import os
import shlex
import shutil
import subprocess
import sys

import numpy as np

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
    if (not os.path.isfile(_LIBRARY_PATH)
            or os.path.getmtime(_LIBRARY_PATH) < newest_source):
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
    """C++ CPU backend compatible with the simulation UI."""

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
