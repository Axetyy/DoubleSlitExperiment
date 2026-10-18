from dataclasses import dataclass
import numpy as np


@dataclass
class Params:
    x_src: int = 60
    x_wall: int = 150
    x_det: int = 480
    
    nx: int = 1000
    ny: int = 600
    view_ny: int = 400

    wall_thickness: int = 4
    sponge: int = 40

    courant: float = 0.4

    wavelength: float = 12.0
    n_slits: int = 2
    slit_width: float = 10.0
    slit_spacing: float = 40.0
    slit_shape: str = "rectangular"

    source_angle_deg: float = 0.0
    source_phase: float = 0.0
    source_y_offset: float = 0.0
    coherent: bool = True
    particle_mode: bool = False

    detector_oversampling: int = 4
    detector_width: int = 4

    wavelength_nm: float = 632.8
    slit_width_um: float = 50.0
    slit_spacing_um: float = 250.0
    screen_distance_m: float = 1.5
    screen_half_height_mm: float = 15.0
    
    um_to_cells: float = 0.18
    cells_per_m: float = 220.0       
    lam_cells_min: float = 8.0       
    lam_cells_max: float = 24.0  
    
    nz:int = 1
    slit_height_um:float = 400.0
    slit_height:float = 0.0
    
def params_3d()->Params:
    p = Params(
        x_src=30,
        x_wall=75,
        nx=400,
        ny=192,
        nz=192,
        wall_thickness=3,
        sponge=12,
        courant=0.5,
        um_to_cells=0.10,
        cells_per_m=80.0,
        lam_cells_min=7.0,
        lam_cells_max=14.0,
    )
    p.x_det = p.x_wall + int(round(p.cells_per_m * p.screen_distance_m))
    p.wavelength = float(np.interp(p.wavelength_nm,[400,700],[p.lam_cells_min,p.lam_cells_max]))
    p.slit_width = p.slit_width_um * p.um_to_cells
    p.slit_height = p.slit_height_um * p.um_to_cells
    return p