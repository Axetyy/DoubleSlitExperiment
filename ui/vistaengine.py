"""PyVista/VTK GPU renderer for the CUDA-aware 3D double-slit simulation."""
import argparse
import sys

import numpy as np
import pyvista as pv
import torch
import vtk

from config import params_3d
from physics.physics3d import TorchWaveSim3D
from reference import fraunhofer3d
from sampling.sampling3d import ParticleSampler3D

RENDER_STRIDE = 3
DRAW_EVERY = 2
VISIBLE_SPONGE_CELLS = 8


def read_params():
    params = params_3d()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelength-nm", type=float, default=params.wavelength_nm)
    parser.add_argument("--slits", type=int, default=params.n_slits)
    parser.add_argument("--slit-width-um", type=float, default=params.slit_width_um)
    parser.add_argument("--slit-spacing-um", type=float, default=params.slit_spacing_um)
    parser.add_argument("--slit-height-um", type=float, default=params.slit_height_um)
    parser.add_argument("--screen-distance-m", type=float, default=params.screen_distance_m)
    parser.add_argument("--source-angle", type=float, default=params.source_angle_deg)
    parser.add_argument("--source-distance", type=int,
                        default=params.x_wall - params.x_src)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--photons", type=int, default=10)
    parser.add_argument("--coherence", choices=("coherent", "incoherent"), default="coherent")
    parser.add_argument("--mode", choices=("wave", "particle"), default="particle")
    parser.add_argument(
        "--slit-shape", choices=("rectangular", "circular"), default="rectangular"
    )
    args = parser.parse_args()

    params.wavelength_nm = args.wavelength_nm
    params.n_slits = args.slits
    params.slit_width_um = args.slit_width_um
    params.slit_spacing_um = args.slit_spacing_um
    params.slit_height_um = args.slit_height_um
    params.screen_distance_m = args.screen_distance_m
    params.source_angle_deg = args.source_angle
    params.x_src = params.x_wall - args.source_distance
    params.x_det = params.x_wall + int(round(params.cells_per_m * params.screen_distance_m))
    params.wavelength = float(np.interp(
        params.wavelength_nm, [400, 700], [params.lam_cells_min, params.lam_cells_max]
    ))
    params.slit_width = params.slit_width_um * params.um_to_cells
    params.slit_spacing = params.slit_spacing_um * params.um_to_cells
    params.slit_height = params.slit_height_um * params.um_to_cells
    params.coherent = args.coherence == "coherent"
    params.particle_mode = args.mode == "particle"
    params.slit_shape = args.slit_shape
    return params, args


p, args = read_params()
device = "cuda" if torch.cuda.is_available() else "cpu"
sim = TorchWaveSim3D(p, device)
sampler = ParticleSampler3D(p.ny, p.nz, p.x_det)
arrival_step = int(np.ceil((p.x_det - p.x_src) / p.courant))
visible_x = min(p.nx, p.x_det + 9)
crop_start = max(0, p.sponge - VISIBLE_SPONGE_CELLS)
crop_y = (crop_start, p.ny - crop_start)
crop_z = (crop_start, p.nz - crop_start)
view_ny = crop_y[1] - crop_y[0]
view_nz = crop_z[1] - crop_z[0]
state = {
    "paused": False,
    "particles": p.particle_mode,
    "wavefronts": True,
    "circular": p.slit_shape == "circular",
    "steps": max(1, args.steps),
    "photons": max(1, args.photons),
}
screen_y_mm = (np.arange(p.ny) + 0.5) * (
    2 * p.screen_half_height_mm / p.ny
) - p.screen_half_height_mm
fraunhofer_profile = fraunhofer3d(
    p,
    np.arange(p.ny),
    sim.slit_open,
    sim.slit_open_z,
    sim.src_y[0],
    sim.src_z,
    sim.aperture_mask,
)

grid_shape = (
    len(range(crop_z[0], crop_z[1], RENDER_STRIDE)),
    len(range(crop_y[0], crop_y[1], RENDER_STRIDE)),
    len(range(0, visible_x, RENDER_STRIDE)),
)
grid = pv.ImageData(
    dimensions=(grid_shape[2], grid_shape[1], grid_shape[0]),
    spacing=(RENDER_STRIDE, RENDER_STRIDE, RENDER_STRIDE),
    origin=(0, 0, 0),
)
grid.point_data["wave"] = np.zeros(np.prod(grid_shape), dtype=np.float32)
wave_scalars = grid.GetPointData().GetScalars()

plotter = pv.Plotter(window_size=(1400, 900), title=f"3D double slit — {sim.label}")
plotter.set_background("#0b0e17")
plotter.add_axes()


def make_isosurface_actor(name, values, color):
    contour = vtk.vtkContourFilter()
    contour.SetInputData(grid)
    contour.SetNumberOfContours(len(values))
    for index, value in enumerate(values):
        contour.SetValue(index, value)
    contour.ComputeNormalsOn()
    contour.Update()
    actor = plotter.add_mesh(
        contour.GetOutputPort(),
        color=color,
        opacity=0.09,
        smooth_shading=False,
        ambient=0.35,
        diffuse=0.65,
        specular=0.12,
        show_scalar_bar=False,
        name=name,
        reset_camera=False,
    )
    return contour, actor


negative_contour, negative_actor = make_isosurface_actor(
    "negative wavefront", (-0.5,), "#55aaff"
)
positive_contour, positive_actor = make_isosurface_actor(
    "positive wavefront", (0.5,), "#ff795e"
)


def make_wall():
    boxes = []
    blocked = ~sim.aperture_mask
    for y in range(crop_y[0], crop_y[1]):
        z_blocked = blocked[crop_z[0]:crop_z[1], y]
        edges = np.diff(np.r_[False, z_blocked, False].astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        for z_start, z_stop in zip(starts, stops):
            boxes.append(pv.Box((
                p.x_wall, p.x_wall + p.wall_thickness,
                y - crop_y[0], y - crop_y[0] + 1,
                z_start, z_stop,
            )))
    if boxes:
        plotter.add_mesh(
            pv.MultiBlock(boxes).combine(),
            color="#747d8e",
            opacity=0.12,
            smooth_shading=False,
            name="slit barrier",
            reset_camera=False,
        )


make_wall()
plotter.add_mesh(
    pv.Sphere(radius=2, center=(
        p.x_src,
        int(sim.src_y[0]) - crop_y[0],
        sim.src_z - crop_z[0],
    )),
    color="#4dff78",
    name="source",
    reset_camera=False,
)
plotter.add_mesh(
    pv.Box((0, visible_x, 0, view_ny, 0, view_nz)),
    style="wireframe",
    color="#69758c",
    line_width=1,
    name="domain",
    reset_camera=False,
)

pixel_y, pixel_z = np.mgrid[crop_y[0]:crop_y[1]:2, crop_z[0]:crop_z[1]:2]
pixel_points = np.column_stack((
    np.full(pixel_y.size, p.x_det + 2.0),
    (pixel_y - crop_y[0]).ravel(),
    (pixel_z - crop_z[0]).ravel(),
))
detector = pv.PolyData(pixel_points)
detector.point_data["intensity"] = np.zeros(pixel_y.size, dtype=np.float32)
detector_actor = plotter.add_points(
    detector,
    scalars="intensity",
    cmap="inferno",
    clim=(0, 1),
    point_size=8,
    render_points_as_spheres=True,
    lighting=False,
    show_scalar_bar=False,
    name="detector pixels",
    reset_camera=False,
)
detector_actor.SetVisibility(False)

status_actor = plotter.add_text(
    f"{sim.label} | starting",
    position=(0.32, 0.97),
    viewport=True,
    font_size=12,
    color="white",
    name="status",
)
plotter.add_text(
    "Blue: negative field   Orange: positive field   White: recent detections",
    position="lower_right",
    font_size=10,
    color="white",
    name="legend",
)
profile_chart = pv.Chart2D()
profile_chart.title = ""
profile_chart.x_axis.label = "screen position (mm)"
profile_chart.y_axis.label = "normalized intensity"
profile_chart.background_color = "#0b0e17"
profile_chart.x_axis.pen.color = "white"
profile_chart.y_axis.pen.color = "white"
profile_chart.x_axis.GetTitleProperties().SetColor(1.0, 1.0, 1.0)
profile_chart.y_axis.GetTitleProperties().SetColor(1.0, 1.0, 1.0)
profile_chart.x_axis.GetLabelProperties().SetColor(1.0, 1.0, 1.0)
profile_chart.y_axis.GetLabelProperties().SetColor(1.0, 1.0, 1.0)
profile_chart.size = (0.26, 0.23)
profile_chart.loc = (0.71, 0.04)
profile_chart.legend_visible = True
profile_chart.x_axis.range = (-p.screen_half_height_mm, p.screen_half_height_mm)
profile_chart.y_axis.range = (0.0, 1.05)
estimate_line = profile_chart.line(
    screen_y_mm,
    fraunhofer_profile,
    color="#ffcf5a",
    width=2,
    style="--",
    label="Fraunhofer estimate",
)
measured_line = profile_chart.line(
    screen_y_mm,
    np.zeros(p.ny),
    color="#67d7ff",
    width=2,
    label="simulation",
)
plotter.add_chart(profile_chart)
plotter.add_text(
    "Detector profile and Fraunhofer estimate",
    position=(0.71, 0.29),
    viewport=True,
    font_size=10,
    color="white",
    name="profile title",
)


def update_detector(image):
    if state["particles"]:
        image = sampler.hist

    reduced = image[crop_y[0]:crop_y[1]:2, crop_z[0]:crop_z[1]:2]
    peak = float(reduced.max())
    arrived = sim.t_step >= arrival_step
    visible = arrived and peak > 0
    if visible:
        detector.point_data["intensity"][:] = (
            reduced / peak
        ).astype(np.float32).ravel()
        detector.GetPointData().GetScalars().Modified()
        detector.Modified()
    detector_actor.SetVisibility(visible)

    if state["particles"] and len(sampler.recent):
        recent = sampler.recent
        points = np.column_stack((
            np.full(len(recent), p.x_det + 2.5),
            recent[:, 0] - crop_y[0],
            recent[:, 1] - crop_z[0],
        ))
        plotter.add_points(
            points,
            color="white",
            point_size=7,
            render_points_as_spheres=True,
            name="recent detections",
            reset_camera=False,
        )
    else:
        plotter.remove_actor("recent detections", render=False)


def update_scene_geometry():
    global arrival_step, visible_x, grid_shape, wave_scalars, fraunhofer_profile

    arrival_step = int(np.ceil((p.x_det - p.x_src) / p.courant))
    visible_x = min(p.nx, p.x_det + 9)
    grid_shape = (
        len(range(crop_z[0], crop_z[1], RENDER_STRIDE)),
        len(range(crop_y[0], crop_y[1], RENDER_STRIDE)),
        len(range(0, visible_x, RENDER_STRIDE)),
    )
    grid.dimensions = (grid_shape[2], grid_shape[1], grid_shape[0])
    grid.point_data["wave"] = np.zeros(np.prod(grid_shape), dtype=np.float32)
    wave_scalars = grid.GetPointData().GetScalars()

    plotter.remove_actor("slit barrier", render=False)
    plotter.remove_actor("source", render=False)
    plotter.remove_actor("domain", render=False)
    make_wall()
    plotter.add_mesh(
        pv.Sphere(radius=2, center=(
            p.x_src,
            int(sim.src_y[0]) - crop_y[0],
            sim.src_z - crop_z[0],
        )),
        color="#4dff78",
        name="source",
        reset_camera=False,
    )
    plotter.add_mesh(
        pv.Box((0, visible_x, 0, view_ny, 0, view_nz)),
        style="wireframe",
        color="#69758c",
        line_width=1,
        name="domain",
        reset_camera=False,
    )
    detector.points[:, 0] = p.x_det + 2.0
    detector.Modified()
    profile_chart.x_axis.range = (-p.screen_half_height_mm, p.screen_half_height_mm)
    fraunhofer_profile = fraunhofer3d(
        p,
        np.arange(p.ny),
        sim.slit_open,
        sim.slit_open_z,
        sim.src_y[0],
        sim.src_z,
        sim.aperture_mask,
    )
    estimate_line.update(screen_y_mm, fraunhofer_profile)
    sampler.reset()
    plotter.reset_camera()
    plotter.camera.zoom(1.5)


def update_spacing_limit():
    slit_count = max(
        1, int(round(parameter_sliders["slits"].GetRepresentation().GetValue()))
    )
    width_cells = (
        parameter_sliders["width"].GetRepresentation().GetValue() * p.um_to_cells
    )
    half_span = p.ny / 2 - p.sponge - width_cells / 2 - 1
    max_spacing = 500 if slit_count <= 1 else min(
        500,
        max(100, half_span / ((slit_count - 1) / 2) / p.um_to_cells),
    )
    spacing_rep = parameter_sliders["spacing"].GetRepresentation()
    spacing_rep.SetMaximumValue(max_spacing)
    if spacing_rep.GetValue() > max_spacing:
        spacing_rep.SetValue(max_spacing)


def update_simulation_parameters(_value=None):
    p.wavelength_nm = parameter_sliders["wavelength"].GetRepresentation().GetValue()
    p.n_slits = int(round(parameter_sliders["slits"].GetRepresentation().GetValue()))
    p.slit_width_um = parameter_sliders["width"].GetRepresentation().GetValue()
    p.slit_spacing_um = parameter_sliders["spacing"].GetRepresentation().GetValue()
    p.slit_height_um = parameter_sliders["height"].GetRepresentation().GetValue()
    p.screen_distance_m = parameter_sliders["distance"].GetRepresentation().GetValue()
    p.source_angle_deg = parameter_sliders["angle"].GetRepresentation().GetValue()
    source_distance = int(round(
        parameter_sliders["source distance"].GetRepresentation().GetValue()
    ))
    p.x_src = p.x_wall - source_distance
    p.x_det = min(
        p.nx - 10,
        p.x_wall + int(round(p.cells_per_m * p.screen_distance_m)),
    )
    p.wavelength = float(np.interp(
        p.wavelength_nm, [400, 700], [p.lam_cells_min, p.lam_cells_max]
    ))
    p.slit_width = p.slit_width_um * p.um_to_cells
    p.slit_spacing = p.slit_spacing_um * p.um_to_cells
    p.slit_height = p.slit_height_um * p.um_to_cells
    p.slit_shape = "circular" if state["circular"] else "rectangular"
    state["steps"] = max(
        1, int(round(parameter_sliders["steps"].GetRepresentation().GetValue()))
    )
    state["photons"] = max(
        1, int(round(parameter_sliders["photons"].GetRepresentation().GetValue()))
    )
    sim.reset()
    update_scene_geometry()
    refresh()


def refresh():
    with torch.no_grad():
        field = sim.u[
            crop_z[0]:crop_z[1]:RENDER_STRIDE,
            crop_y[0]:crop_y[1]:RENDER_STRIDE,
            :visible_x:RENDER_STRIDE,
        ]
        first_wave_x = (p.x_src + 4) // RENDER_STRIDE
        local_peak = field.abs().amax(dim=(0, 1), keepdim=True).clamp_min(1e-7)
        reference_peak = field[:, :, first_wave_x:].abs().amax()
        active_slices = local_peak > reference_peak * 0.001
        positive_peak = field.clamp_min(0).amax(dim=(0, 1), keepdim=True)
        negative_peak = (-field.clamp_max(0)).amax(dim=(0, 1), keepdim=True)
        positive_scale = torch.where(positive_peak > local_peak * 0.02,
                                     positive_peak, local_peak)
        negative_scale = torch.where(negative_peak > local_peak * 0.02,
                                     negative_peak, local_peak)
        values = torch.where(
            field >= 0,
            field / positive_scale,
            field / negative_scale,
        )
        values = torch.where(active_slices, values, 0.0).clamp(-1.0, 1.0)
        values[:, :, :first_wave_x] = 0
        values = values.contiguous().cpu().numpy().ravel(order="C")

    grid.point_data["wave"][:] = values
    wave_scalars.Modified()
    grid.Modified()
    negative_contour.Modified()
    positive_contour.Modified()
    negative_actor.SetVisibility(state["wavefronts"])
    positive_actor.SetVisibility(state["wavefronts"])

    detector_image = sim.detector_image()
    update_detector(detector_image)
    if state["particles"]:
        profile = sampler.hist.sum(axis=1).astype(np.float64)
    else:
        profile = detector_image[:, p.nz // 2]
    profile_peak = float(profile.max())
    normalized_profile = profile / max(profile_peak, 1e-12)
    measured_line.update(screen_y_mm, normalized_profile)

    if sim.t_step < arrival_step:
        message = f"wave travelling to detector: {sim.t_step}/{arrival_step} steps"
    elif state["particles"]:
        message = f"detector accumulated {int(sampler.hist.sum())} hits"
    else:
        message = f"detector intensity peak {float(sim.detector_image().max()):.3g}"
    if state["paused"]:
        message = "paused | " + message
    status_actor.SetInput(f"{sim.label}\n{message}")
    status_actor.Modified()
    plotter.render()


def timer_callback(_step):
    if not state["paused"]:
        sim.step_n(state["steps"])
        if state["particles"] and sim.t_step >= arrival_step:
            sampler.sample(sim.detector_image(), state["photons"])
    if _step % DRAW_EVERY == 0:
        refresh()


def checkbox(label, key, position):
    def changed(value):
        state[key] = bool(value)
        if key == "circular":
            update_simulation_parameters()

    plotter.add_checkbox_button_widget(
        changed,
        value=state[key],
        position=position,
        size=24,
        color_on="#47b8ff",
        color_off="#555b68",
        background_color="#202530",
    )
    plotter.add_text(
        label,
        position=(position[0] + 32, position[1] + 4),
        font_size=10,
        color="white",
        name=f"{key} label",
    )


checkbox("Wavefronts", "wavefronts", (10, 10))
checkbox("Accumulate detections", "particles", (10, 45))
checkbox("Pause", "paused", (10, 80))
checkbox("Circular apertures", "circular", (10, 115))

parameter_sliders = {}
controls_ready = False


def on_parameter_slider(key, value):
    if not controls_ready:
        return
    if key == "steps":
        state["steps"] = max(1, int(round(value)))
        return
    if key == "photons":
        state["photons"] = max(1, int(round(value)))
        return
    if key in {"slits", "width"}:
        update_spacing_limit()
    update_simulation_parameters()


def add_parameter_slider(key, title, value, limits, y, fmt="%.1f", column="left"):
    x0, x1 = (0.03, 0.28) if column == "left" else (0.71, 0.96)
    parameter_sliders[key] = plotter.add_slider_widget(
        lambda new_value, slider_key=key: on_parameter_slider(slider_key, new_value),
        limits,
        value=value,
        title=title,
        pointa=(x0, y),
        pointb=(x1, y),
        style="modern",
        title_color="white",
        interaction_event="end",
        fmt=fmt,
    )


left_ys = (0.93, 0.80, 0.67, 0.54, 0.41)
right_ys = left_ys
add_parameter_slider("wavelength", "wavelength (nm)", p.wavelength_nm,
                     (400, 700), left_ys[0], "%.0f")
add_parameter_slider("slits", "number of slits", p.n_slits,
                     (1, 7), left_ys[1], "%.0f")
add_parameter_slider("width", "slit width / circle diameter (um)", p.slit_width_um,
                     (10, 80), left_ys[2], "%.0f")
spacing_max = min(500, max(100, (
    p.ny / 2 - p.sponge - p.slit_width / 2 - 1
) / max((p.n_slits - 1) / 2, 1) / p.um_to_cells))
add_parameter_slider("spacing", "slit pitch (um)", p.slit_spacing_um,
                     (100, spacing_max), left_ys[3], "%.0f")
add_parameter_slider("height", "slit height (um, rectangular)", p.slit_height_um,
                     (100, 1200), left_ys[4], "%.0f")
add_parameter_slider("distance", "screen distance (m)", p.screen_distance_m,
                     (0.5, 3.5), right_ys[0], "%.1f", "right")
add_parameter_slider("angle", "incidence angle (deg)", p.source_angle_deg,
                     (-5, 5), right_ys[1], "%.1f", "right")
add_parameter_slider("source distance", "source distance (cells)",
                     p.x_wall - p.x_src, (p.sponge, p.x_wall - p.sponge),
                     right_ys[2], "%.0f", "right")
add_parameter_slider("steps", "steps per update", state["steps"], (1, 30),
                     right_ys[3], "%.0f", "right")
add_parameter_slider("photons", "mean detections/update", state["photons"],
                     (1, 100), right_ys[4], "%.0f", "right")
controls_ready = True
plotter.add_key_event("space", lambda: state.__setitem__("paused", not state["paused"]))
plotter.add_key_event("p", lambda: state.__setitem__("paused", not state["paused"]))
plotter.add_key_event("a", lambda: state.__setitem__("particles", not state["particles"]))
plotter.add_key_event("w", lambda: state.__setitem__("wavefronts", not state["wavefronts"]))
plotter.add_key_event("r", lambda: (sim.reset(), sampler.reset()))
plotter.view_isometric()
plotter.reset_camera()
plotter.camera.zoom(1.5)


def main():
    print(
        f"3D solver: {sim.label} ({sim.device}); VTK renders wavefront geometry. "
        f"Configuration: {p.wavelength_nm:g} nm, {p.n_slits} slits, "
        f"{p.screen_distance_m:g} m detector distance."
    )
    if sim.device.type != "cuda":
        print("CUDA is unavailable; simulation is using the PyTorch CPU backend.")

    sim.step_n(80)
    refresh()
    plotter.iren.initialize()
    plotter.add_timer_event(
        max_steps=sys.maxsize,
        duration=30,
        callback=timer_callback,
    )
    plotter.show()


if __name__ == "__main__":
    main()
