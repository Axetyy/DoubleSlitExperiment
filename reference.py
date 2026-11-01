from config import Params
import numpy as np


def _numerical_wavenumber(p: Params, angle:float)->np.ndarray:
    """
    Lattice wavenumber along a direction with angle ( radians ) from the x-asis. Solves sin^2(w dt/2)/C^2 = sin^2(k cos(angle)/2) + sin^2(k sin(angle)/2).
    
    :param p: simulation parameters
    :param angle: angle from the x-axis in radians
    :return: numerical wavenumber
    
    """
    
    omega_dt = 2 * np.pi * p.courant / p.wavelength
    target = np.sin(omega_dt / 2) ** 2 / p.courant ** 2
    angle = np.asarray(angle, dtype=float)
    low = np.zeros_like(angle, dtype=float)
    high = np.full_like(angle, np.pi, dtype=float)

    for _ in range(32):
        wave_number = (low + high) / 2
        dispersion = (
            np.sin(wave_number * np.cos(angle) / 2) ** 2
            + np.sin(wave_number * np.sin(angle) / 2) ** 2
        )
        below_target = dispersion < target
        low = np.where(below_target, wave_number, low)
        high = np.where(below_target, high, wave_number)

    return (low + high) / 2


def fraunhofer(p: Params, detector_rows:np.ndarray, slit_open:np.ndarray, source_row:float)->np.ndarray:
    """
    Calculates the fraunhofer diffraction pattern for a set of given parameters
    
    :param p: simulation parameters
    :param detector_rows: rows of the detector
    :param slit_open: boolean array indicating which rows of the slit are open
    :param source_row: row of the source
    
    :return: normalized intensity along the detector rows
    """
    
    center = p.ny / 2
    detector_y = np.asarray(detector_rows, dtype=float) - center

    idx = np.flatnonzero(np.asarray(slit_open, dtype=bool))
    aperture_y = idx.astype(float) - center

    source_x = p.x_wall + p.wall_thickness / 2 - p.x_src
    screen_x = p.x_det - p.x_wall - p.wall_thickness / 2
    source_y = float(source_row) - center

    incident_angle = np.arctan2(aperture_y - source_y, source_x)
    incident_k = _numerical_wavenumber(p, incident_angle)
    source_distance = np.hypot(source_x, aperture_y - source_y)
    aperture_field = np.exp(1j * incident_k * source_distance) / np.sqrt(source_distance)

    sin_theta = detector_y / np.hypot(detector_y, screen_x)
    detector_angle = np.arctan2(detector_y, screen_x)
    detector_k = _numerical_wavenumber(p, detector_angle)
    propagation = np.exp(
        -1j * detector_k[:, None] * sin_theta[:, None] * aperture_y[None, :]
    )
    intensity = np.abs(propagation @ aperture_field) ** 2

    peak = float(np.max(intensity)) if intensity.size else 0.0
    return intensity / peak if peak > 0 else intensity


def _numerical_wavenumber_3d(p:Params, cx:np.ndarray, cy:np.ndarray, cz:np.ndarray)->np.ndarray:
    """Lattice wavenumber along a direction with cosines (cx, cy, cz).

    Solves  sin^2(w dt/2)/C^2 = sin^2(k cx/2) + sin^2(k cy/2) + sin^2(k cz/2).
    
    :param p: simulation parameters
    :param cx : cosine of the angle in the x-direction
    :param cy : cosine of the angle in the y-direction
    :param cz : cosine of the angle in the z-direction
    
    :return: numerical wavenumber
    """
    omega_dt = 2 * np.pi * p.courant / p.wavelength
    target = np.sin(omega_dt / 2) ** 2 / p.courant ** 2
    cx, cy, cz = np.broadcast_arrays(
        np.asarray(cx, float), np.asarray(cy, float), np.asarray(cz, float)
    )
    low = np.zeros_like(cx)
    high = np.full_like(cx, np.pi)
 
    for _ in range(32):
        k = (low + high) / 2
        f = (
            np.sin(k * cx / 2) ** 2
            + np.sin(k * cy / 2) ** 2
            + np.sin(k * cz / 2) ** 2
        )
        below = f < target
        low = np.where(below, k, low)
        high = np.where(below, high, k)
 
    return (low + high) / 2
 
 
def fraunhofer3d(
    p:Params, detector_rows:np.ndarray, slit_open_y:np.ndarray, slit_open_z:np.ndarray, source_y:float, source_z:float, aperture_mask=None
):
    """
        Calculates the fraunhofer diffraction pattern for a set of given parameters in 3D.
        
        :param p: simulation parameters
        :param detector_rows: rows of the detector
        :param slit_open_y: boolean array indicating which rows of the slit are open in the y-direction
        :param slit_open_z: boolean array indicating which rows of the slit are open in the z-direction
        :param source_y: y-coordinate of the source
        :param source_z: z-coordinate of the source
        
        :return: normalized intensity along the detector rows
        """
    cy0, cz0 = p.ny / 2, p.nz / 2
    det_y = np.asarray(detector_rows, dtype=float) - cy0
 
    if aperture_mask is None:
        aperture = (
            np.asarray(slit_open_z, dtype=bool)[:, None]
            & np.asarray(slit_open_y, dtype=bool)[None, :]
        )
    else:
        aperture = np.asarray(aperture_mask, dtype=bool)
    iz, iy = np.nonzero(aperture)
    if iy.size == 0:
        return np.zeros_like(det_y)

    ay_all = np.unique(iy).astype(float) - cy0
    aperture_y = iy.astype(float) - cy0
    aperture_z = iz.astype(float) - cz0
 
    sx = p.x_wall + p.wall_thickness / 2 - p.x_src
    scr = p.x_det - p.x_wall - p.wall_thickness / 2
    sy = float(source_y) - cy0
    sz = float(source_z) - cz0
 
    dy = aperture_y - sy
    dz = aperture_z - sz
    r = np.sqrt(sx ** 2 + dy ** 2 + dz ** 2)
    k_in = _numerical_wavenumber_3d(p, sx / r, dy / r, dz / r)
    cell_fields = np.exp(1j * k_in * r) / r
    summed_field = np.bincount(iy, weights=cell_fields.real, minlength=p.ny)
    summed_field = summed_field + 1j * np.bincount(
        iy, weights=cell_fields.imag, minlength=p.ny
    )
    aperture_field = summed_field[np.unique(iy)]

    R = np.hypot(det_y, scr)
    sin_t = det_y / R
    k_out = _numerical_wavenumber_3d(p, scr / R, det_y / R, np.zeros_like(R))
    propagation = np.exp(-1j * k_out[:, None] * sin_t[:, None] * ay_all[None, :])
    intensity = np.abs(propagation @ aperture_field) ** 2
 
    peak = float(np.max(intensity)) if intensity.size else 0.0
    return intensity / peak if peak > 0 else intensity