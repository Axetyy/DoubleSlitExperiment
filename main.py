import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, RadioButtons

from config import Params, params_3d
from reference import fraunhofer, fraunhofer3d
from sampling.sampling import ParticleSampler
from sampling.sampling3d import ParticleSampler3D

from utils import wavelength_to_rgb, high_res_screen, runs
from ui.ui import make_slider, quiet_set, retarget, radio_quiet

try:
    from physics.core import WaveSimulator as NumbaSim2D
except Exception:
    NumbaSim2D = None

try:
    from physics.core import TorchWaveSim2D, cuda_available
except Exception:
    TorchWaveSim2D = None
    def cuda_available(): return False

try:
    from physics.core import CppWaveSim2D
except Exception:
    CppWaveSim2D = None

try:
    import physics.core as physics3d
except Exception:
    physics3d = None
DIM = 2
DEV = "CPU"
p2 = Params()
p3 = params_3d()
p = p2
sim = None
sampler = None
backend_label = ""
state = {"paused": False, "scale": 1.0}

screen_arrival_step = 0
screen_pixel_size_mm = 1.0
screen_y_mm = np.zeros(1)
screen_z_half = 1.0
screen_rows = np.arange(p.ny)
fraunhofer_profile = np.zeros(p.ny)
slice_surf = None
screen_surf = None
wall_bars = []
SURF_STEP = 4     
SURF_EVERY = 2  

# --- Figure & Layout Setup ---
fig = plt.figure(figsize=(11, 8.8))
gs = GridSpec(1, 3, width_ratios=[4, 0.3, 1.1], left=0.05, right=0.98, top=0.95,
              bottom=0.40, wspace=0.03)
ax_f = fig.add_subplot(gs[0])
ax_s = fig.add_subplot(gs[1])
ax_p = fig.add_subplot(gs[2])
ax_f3 = fig.add_subplot(gs[0], projection="3d")
ax_f3.set_visible(False)
POS2 = {"f": ax_f.get_position().frozen(), "s": ax_s.get_position().frozen(),
        "p": ax_p.get_position().frozen()}

im = ax_f.imshow(np.zeros((min(p2.ny, p2.view_ny), p2.nx)), origin="lower",
                 cmap="coolwarm", vmin=-1, vmax=1,
                 interpolation="nearest", aspect="auto")
wall_im = ax_f.imshow(np.zeros((min(p2.ny, p2.view_ny), p2.nx, 4), np.uint8),
                      origin="lower", interpolation="nearest", aspect="auto")
src_pts, = ax_f.plot([], [], "g.", ms=3)
detector_line = ax_f.axvline(p2.x_det, color="k", lw=2)
hit_sc = ax_f.scatter([], [], s=4, c="darkorange", edgecolors="none", visible=False)
status = ax_f.text(0.02, 0.96, "", transform=ax_f.transAxes, fontsize=9,
                   bbox=dict(fc="white", ec="none", alpha=0.8))
ax_f.set_autoscale_on(False) 

status3 = ax_f3.text2D(0.02, 0.96, "", transform=ax_f3.transAxes, fontsize=9,
                       bbox=dict(fc="white", ec="none", alpha=0.8))
src3d, = ax_f3.plot([0], [0], [0], "g.", ms=8)
ax_f3.set_xlabel("x", fontsize=8)
ax_f3.set_ylabel("y", fontsize=8)
ax_f3.set_zlabel("z", fontsize=8)
ax_f3.view_init(elev=30, azim=-65)
 
# Detector strip & profile setups
strip = ax_s.imshow(np.zeros((p2.ny * p2.detector_oversampling, 1, 3), dtype=np.float32),
                    origin="lower",
                    extent=(0, 1, -p2.screen_half_height_mm, p2.screen_half_height_mm),
                    vmin=0, vmax=1, aspect="auto", interpolation="nearest")
screen_hit_sc = ax_s.scatter([], [], s=10, c="darkorange", edgecolors="white",
                             linewidths=0.35, zorder=3, visible=False)
ax_s.set_ylabel("position (mm)", fontsize=8)
 
line_I, = ax_p.plot([], [], "b", lw=1.4, label="wave intensity", visible=True)
line_H, = ax_p.plot([], [], "k", lw=1, drawstyle="steps-mid", label="cumulative detections",
                    visible=False)
line_F, = ax_p.plot([], [], "r--", lw=1.2, label="grid Fraunhofer approximation", visible=True)
ax_p.set_autoscale_on(False)
ax_p.set_xlim(0, 1)
ax_p.set_title("screen profile", fontsize=9)
ax_p.grid(True, alpha=0.25)
ax_p.set_xlabel("normalized intensity")
ax_p.set_ylabel("position (mm)")
ax_p.legend(handles=[line_I, line_F], loc="upper right", fontsize=7)
 
note = fig.text(0.5, 0.005, "", ha="center", fontsize=8, color="crimson")

def say(msg):
    note.set_text(msg) 
 
def set_status(msg):
    status.set_text(msg)
    status3.set_text(msg)

def get_field():
    return sim.field_slice() if hasattr(sim, "field_slice") else sim.u
 
def det_profile():
    return sim.detector_profile() if hasattr(sim, "detector_profile") else sim.I_det
 
def det_image():
    return sim.detector_image()

screen_color = wavelength_to_rgb(p.wavelength_nm)

# Sliders & Radio Buttons
ROWS = [0.30, 0.24, 0.18, 0.12, 0.06]
X1, X2, W = 0.31, 0.74, 0.19
s_lam = make_slider(fig, [X1, ROWS[0], W, 0.03], "wavelength (nm)", 400, 700, p2.wavelength_nm, 0.1)
s_n = make_slider(fig, [X1, ROWS[1], W, 0.03], "# slits", 1, 7, p2.n_slits, 1)
s_a = make_slider(fig, [X1, ROWS[2], W, 0.03], "slit width (μm)", 10, 80, p2.slit_width_um, 1)
s_d = make_slider(fig, [X1, ROWS[3], W, 0.03], "slit pitch (μm)", 100, 500, p2.slit_spacing_um, 5)
s_src = make_slider(fig, [X1, ROWS[4], W, 0.03], "source dist (cells)", 10, 110, p2.x_wall - p2.x_src, 1)
s_L = make_slider(fig, [X2, ROWS[0], W, 0.03], "screen L (m)", 0.5, 3.5, p2.screen_distance_m, 0.1)
s_ang = make_slider(fig, [X2, ROWS[1], W, 0.03], "incidence (deg)", -5, 5, p2.source_angle_deg, 0.1)
s_spf = make_slider(fig, [X2, ROWS[2], W, 0.03], "steps/frame", 1, 30, 10, 1)
s_ppf = make_slider(fig, [X2, ROWS[3], W, 0.03], "mean photons/frame", 1, 100, 10, 1)
s_h = make_slider(fig, [X2, ROWS[4], W, 0.03], "slit height (μm) [3D]", 100, 1200, p3.slit_height_um, 10)
 
rb_dim = RadioButtons(fig.add_axes([0.005, 0.27, 0.085, 0.07]), ["2D", "3D"])
rb_dev = RadioButtons(fig.add_axes([0.005, 0.18, 0.085, 0.085]), ["CPU", "C++", "GPU"])
rb_mode = RadioButtons(fig.add_axes([0.005, 0.11, 0.085, 0.07]), ["Wave", "Particle"])
rb_coh = RadioButtons(fig.add_axes([0.005, 0.03, 0.085, 0.07]), ["Coherent", "Incoherent"])
for rb in (rb_dim, rb_dev, rb_mode, rb_coh):
    for lab in rb.labels:
        lab.set_fontsize(8)

b_reset = Button(fig.add_axes([0.10, 0.27, 0.07, 0.04]), "Reset")
b_pause = Button(fig.add_axes([0.10, 0.20, 0.07, 0.04]), "Pause")

# Backend import handling inside main



def configure_geometry():
    global screen_pixel_size_mm, screen_y_mm, screen_z_half, screen_rows
    global screen_arrival_step, screen_color

    half = p.screen_half_height_mm
    view_ny = min(p.ny, p.view_ny) if DIM == 2 else p.ny
    view_start = (p.ny - view_ny) // 2
    screen_rows = np.arange(view_start, view_start + view_ny)
    screen_pixel_size_mm = 2 * half / view_ny
    screen_y_mm = (np.arange(view_ny) + 0.5) * screen_pixel_size_mm - half
    screen_z_half = p.nz * screen_pixel_size_mm / 2
    screen_color = wavelength_to_rgb(p.wavelength_nm)
    screen_arrival_step = int(np.ceil((p.x_det - p.x_src) / p.courant))
 
    quiet_set(s_lam, p.wavelength_nm)
    quiet_set(s_n, p.n_slits)
    quiet_set(s_a, p.slit_width_um)
    quiet_set(s_d, p.slit_spacing_um)
    quiet_set(s_L, p.screen_distance_m)
    quiet_set(s_ang, p.source_angle_deg)
    lo = 40 if DIM == 2 else 15
    retarget(s_src, lo, p.x_wall - p.sponge, p.x_wall - p.x_src)
    quiet_set(s_h, p.slit_height_um)
    s_h.ax.set_visible(DIM == 3)
    set_layout()

def set_layout():
    half = p.screen_half_height_mm
    if DIM == 2:
        ax_f.set_visible(True)
        ax_f3.set_visible(False)
        ax_f.set_position(POS2["f"])
        ax_s.set_position(POS2["s"])
        ax_p.set_position(POS2["p"])
        ax_s.set_aspect("auto", adjustable="box")
        strip.set_extent((0, 1, -half, half))
        ax_s.set_xlim(0, 1)
        ax_s.set_ylim(-half, half)
        ax_s.set_xticks([])
        ax_s.set_xlabel("")
        ax_s.set_title("detector", fontsize=8)
        ax_f.set_xlim(-0.5, p.x_det + 8.5)
        ax_f.set_ylim(screen_rows[0] - 0.5, screen_rows[-1] + 0.5)
        view_extent = (-0.5, p.nx - 0.5, screen_rows[0] - 0.5, screen_rows[-1] + 0.5)
        im.set_extent(view_extent)
        wall_im.set_extent(view_extent)
        detector_line.set_xdata([p.x_det, p.x_det])
    else:
        zh = screen_z_half
        ax_f.set_visible(False)
        ax_f3.set_visible(True)
        ax_f3.set_position([0.02, 0.38, 0.52, 0.60])
        ax_s.set_position([0.58, 0.45, 0.22, 0.45])
        ax_p.set_position([0.86, 0.45, 0.12, 0.45])
        ax_s.set_aspect("equal", adjustable="box")
        strip.set_extent((-zh, zh, -half, half))
        ax_s.set_xlim(-zh, zh)
        ax_s.set_ylim(-half, half)
        ax_s.set_xticks([-zh, 0, zh])
        ax_s.set_xticklabels([f"{-zh:.1f}", "0", f"{zh:.1f}"], fontsize=7)
        ax_s.set_xlabel("z (mm)", fontsize=8)
        ax_s.set_title("detector plane", fontsize=8)
        set_3d_limits()
    ax_p.set_ylim(-half, half)
    ax_f.set_title(f"qualitative wave field (grid units) - {backend_label}", fontsize=9)
    ax_f3.set_title(f"z = centre slice, wall, detector plane - {backend_label}", fontsize=9)

def set_3d_limits():
    xmax = p.x_det + 8
    ax_f3.set_xlim(0, xmax)
    ax_f3.set_ylim(0, p.ny)
    ax_f3.set_zlim(0, p.nz)
    ax_f3.set_box_aspect((xmax, p.ny, p.nz))

def build_sim():
    global sim, sampler, backend_label, DEV
    dev = DEV
    if dev == "GPU" and not cuda_available():
        say("CUDA GPU not available - running on CPU instead.")
        dev = "CPU"
        DEV = "CPU"
        radio_quiet(rb_dev, 0)
    else:
        say("")
 
    if DIM == 2:
        if dev == "GPU":
            sim = TorchWaveSim2D(p, "cuda")
        elif dev == "C++":
            if CppWaveSim2D is None:
                raise RuntimeError("C++ backend unavailable.")
            sim = CppWaveSim2D(p)
        elif NumbaSim2D is not None:
            sim = NumbaSim2D(p)
            sim.label = "CPU numba"
        elif TorchWaveSim2D is not None:
            sim = TorchWaveSim2D(p, "cpu")
        else:
            raise RuntimeError("Install numba or torch.")
        sampler = ParticleSampler(p.ny, p.x_det)
    else:
        if physics3d is None:
            raise RuntimeError("physics3d unavailable.")
        if dev == "GPU":
            sim = physics3d.TorchWaveSim3D(p, "cuda")
        elif physics3d.HAVE_NUMBA:
            sim = physics3d.WaveSim3D(p)
        else:
            sim = physics3d.TorchWaveSim3D(p, "cpu")
        sampler = ParticleSampler3D(p.ny, p.nz, p.x_det)
 
    backend_label = f"{DIM}D | {getattr(sim, 'label', 'CPU')}"

def clear_particles():
    sampler.reset()

def refresh_static():
    global fraunhofer_profile, wall_bars
    if DIM == 2:
        rgba = np.zeros((len(screen_rows), p.nx, 4), np.uint8)
        rgba[~sim.slit_open[screen_rows], p.x_wall:p.x_wall + p.wall_thickness] = (25, 25, 25, 255)
        wall_im.set_data(rgba)
        src_pts.set_data(np.full(len(sim.src_y), p.x_src), sim.src_y)
        fraunhofer_profile = fraunhofer(p, screen_rows, sim.slit_open, sim.src_y[0])
    else:
        for art in wall_bars:
            art.remove()
        wall_bars = []
        zo = np.flatnonzero(sim.slit_open_z)
        z_lo, z_hi = (int(zo.min()), int(zo.max()) + 1) if zo.size else (p.nz // 2, p.nz // 2)
        blocks = []
        for y0, y1, is_open in runs(sim.slit_open):
            if not is_open:
                blocks.append((y0, y1, 0, p.nz))
            else:
                if z_lo > 0:
                    blocks.append((y0, y1, 0, z_lo))
                if z_hi < p.nz:
                    blocks.append((y0, y1, z_hi, p.nz))
        for y0, y1, z0, z1 in blocks:
            wall_bars.append(ax_f3.bar3d(p.x_wall, y0, z0, p.wall_thickness, y1 - y0, z1 - z0,
                                         color="0.25", alpha=0.55, shade=True))
        src3d.set_data_3d([p.x_src], [int(sim.src_y[0])], [sim.src_z])
        fraunhofer_profile = fraunhofer3d(p, screen_rows, sim.slit_open, sim.slit_open_z,
                                          sim.src_y[0], sim.src_z)
    line_F.set_data(fraunhofer_profile, screen_y_mm)

def apply_mode_visuals():
    pm = p.particle_mode
    im.set_alpha(0.3 if pm else 1.0)
    hit_sc.set_visible(pm and DIM == 2)
    screen_hit_sc.set_visible(pm)
    line_I.set_visible(not pm)
    line_H.set_visible(False)
    line_F.set_label("Fraunhofer expected counts" if pm else "grid Fraunhofer approximation")
    ax_p.set_xlabel("detections per screen bin" if pm else "normalized wave intensity")
    ax_p.set_title("cumulative photon detections" if pm else "simulated detector intensity")
    ax_p.set_xlim(0, 1.05)
    ax_p.legend(handles=[line_H, line_F] if pm else [line_I, line_F], loc="upper right", fontsize=7)
    clear_particles()

def activate():
    global p, slice_surf, screen_surf
    p = p3 if DIM == 3 else p2
    p.coherent = rb_coh.value_selected == "Coherent"
    p.particle_mode = rb_mode.value_selected == "Particle"
 
    for art in (slice_surf, screen_surf):
        if art is not None:
            art.remove()
    slice_surf = screen_surf = None
 
    build_sim()
    configure_geometry()
    refresh_static()
    apply_mode_visuals()
    state["scale"] = 1.0

# Callbacks and Event Handlers
def on_param(_=None):
    global screen_arrival_step, screen_color
    p.wavelength_nm = s_lam.val
    p.slit_width_um = s_a.val
    p.slit_spacing_um = s_d.val
    p.slit_height_um = s_h.val
    p.screen_distance_m = s_L.val
    p.x_src = p.x_wall - int(s_src.val)
    p.x_det = p.x_wall + int(round(p.cells_per_m * p.screen_distance_m))
    p.source_angle_deg = s_ang.val
 
    p.wavelength = float(np.interp(p.wavelength_nm, [400, 700], [p.lam_cells_min, p.lam_cells_max]))
    screen_color = wavelength_to_rgb(p.wavelength_nm)
    p.n_slits = int(s_n.val)
    p.slit_width = p.slit_width_um * p.um_to_cells
    p.slit_spacing = p.slit_spacing_um * p.um_to_cells
    p.slit_height = p.slit_height_um * p.um_to_cells
    screen_arrival_step = int(np.ceil((p.x_det - p.x_src) / p.courant))
    sampler.x_detector = p.x_det
 
    if DIM == 2:
        detector_line.set_xdata([p.x_det, p.x_det])
        ax_f.set_xlim(-0.5, p.x_det + 8.5)
    else:
        set_3d_limits()
    sim.reset()
    refresh_static()
    clear_particles()

def on_dim(label):
    global DIM, DEV
    new = 3 if label == "3D" else 2
    if new == DIM:
        return
    if new == 3 and DEV == "GPU" and cuda_available():
        radio_quiet(rb_dim, 0)
        return
    DIM = new
    activate()

def on_dev(label):
    global DEV
    previous = DEV
    DEV = label
    try:
        build_sim()
    except Exception as exc:
        DEV = previous
        radio_quiet(rb_dev, ["CPU", "C++", "GPU"].index(previous))
        say(f"Could not start {label}: {exc}")
        return
    set_layout()
    refresh_static()
    clear_particles()

def on_mode(label):
    p.particle_mode = label == "Particle"
    apply_mode_visuals()

def on_coherence(label):
    p.coherent = label == "Coherent"
    sim.reset()
    refresh_static()
    clear_particles()

def on_reset(_):
    sim.reset()
    clear_particles()

def on_pause(_):
    state["paused"] = not state["paused"]
    b_pause.label.set_text("Resume" if state["paused"] else "Pause")

for s in (s_lam, s_n, s_a, s_d, s_L, s_ang, s_src, s_h):
    s.on_changed(on_param)
rb_dim.on_clicked(on_dim)
rb_dev.on_clicked(on_dev)
rb_mode.on_clicked(on_mode)
rb_coh.on_clicked(on_coherence)
b_reset.on_clicked(on_reset)
b_pause.on_clicked(on_pause)

def draw_3d(disp):
    global slice_surf, screen_surf
    cmap = plt.get_cmap("coolwarm")
    xs = min(p.nx, p.x_det + 9)
    sub = disp[::SURF_STEP, :xs:SURF_STEP]
    rgba = cmap(np.clip(0.5 + 0.5 * sub / max(state["scale"], 1e-6), 0, 1))
    xi = np.arange(0, xs, SURF_STEP)[: sub.shape[1]]
    yi = np.arange(0, p.ny, SURF_STEP)[: sub.shape[0]]
    X, Y = np.meshgrid(xi, yi)
    Z = np.full(X.shape, p.nz / 2.0)
    if slice_surf is not None:
        slice_surf.remove()
    slice_surf = ax_f3.plot_surface(X, Y, Z, facecolors=rgba, rstride=1, cstride=1,
                                    shade=False, linewidth=0, antialiased=False)
 
    if p.particle_mode:
        hist = sampler.hist
        colors = plt.get_cmap("inferno")(hist / max(float(hist.max()), 1.0))
    else:
        img = det_image()
        img = img / max(float(img.max()), 1e-12)
        colors = np.ones(img.shape + (4,))
        colors[..., :3] = img[..., None] * screen_color[None, None, :]
    colors = colors[::SURF_STEP, ::SURF_STEP]
    yi = np.arange(0, p.ny, SURF_STEP)[: colors.shape[0]]
    zi = np.arange(0, p.nz, SURF_STEP)[: colors.shape[1]]
    Yg, Zg = np.meshgrid(yi, zi, indexing="ij")
    Xg = np.full(Yg.shape, float(p.x_det))
    if screen_surf is not None:
        screen_surf.remove()
    screen_surf = ax_f3.plot_surface(Xg, Yg, Zg, facecolors=colors, rstride=1, cstride=1,
                                     shade=False, linewidth=0, antialiased=False)

def update(frame):
    if not state["paused"]:
        sim.step_n(int(s_spf.val))
        if p.particle_mode and sim.t_step >= screen_arrival_step:
            sampler.sample(det_image() if DIM == 3 else det_profile(), int(s_ppf.val))
 
    field = get_field()
    disp = np.sign(field) * np.sqrt(np.abs(field))
    if frame % 5 == 0:
        region = disp[::2, p.x_src + 20::2]
        if region.size:
            target = max(float(np.percentile(np.abs(region), 99.5)), 0.02)
            state["scale"] = 0.8 * state["scale"] + 0.2 * target
    if DIM == 2:
        im.set_data(disp[screen_rows])
        im.set_clim(-state["scale"], state["scale"])
    elif frame % SURF_EVERY == 0:
        draw_3d(disp)
 
    arrived = sim.t_step >= screen_arrival_step
    half = p.screen_half_height_mm
 
    if p.particle_mode:
        hist = sampler.hist
        counts = hist[screen_rows] if DIM == 2 else hist.sum(axis=1)
        max_count = float(counts.max())
        total_count = float(counts.sum())
        line_H.set_data(counts, screen_y_mm)
        line_H.set_visible(max_count > 0)
        expected_counts = (fraunhofer_profile * total_count / max(fraunhofer_profile.sum(), 1e-12))
        line_F.set_data(expected_counts, screen_y_mm)
        ax_p.set_xlim(0, max(1, max(max_count, float(expected_counts.max())) * 1.1))
        if DIM == 2:
            hit_sc.set_offsets(sampler.recent)
            screen_hit_sc.set_offsets(np.column_space((
                np.full(len(sampler.recent), 0.5),
                (sampler.recent[:, 1] - screen_rows[0] + 0.5) * screen_pixel_size_mm - half,
            )) if len(sampler.recent) > 0 else np.empty((0, 2)))
            screen_hits = high_res_screen(counts, p.detector_oversampling)
            strip.set_data(screen_hits[:, None])
            strip.set_clim(0, max(1, max_count))
        else:
            rec = sampler.recent
            offsets = np.column_stack((
                (rec[:, 1] + 0.5) * screen_pixel_size_mm - screen_z_half,
                (rec[:, 0] + 0.5) * screen_pixel_size_mm - half,
            )) if len(rec) > 0 else np.empty((0, 2))
            screen_hit_sc.set_offsets(offsets)
            strip.set_data(hist)
            strip.set_clim(0, max(1.0, float(hist.max())))
        if sim.t_step < screen_arrival_step:
            set_status("wave travelling to screen...")
        elif total_count == 0:
            set_status("wave reached screen - accumulating detections...")
        else:
            set_status(f"{int(total_count)} detections")
    else:
        profile = det_profile()
        if DIM == 2:
            profile = profile[screen_rows]
        detector_peak = float(profile.max())
        wave_profile = profile / max(detector_peak, 1e-12)
        line_I.set_data(wave_profile, screen_y_mm)
        line_I.set_visible(arrived and detector_peak > 1e-12)
        if DIM == 2:
            screen_I = high_res_screen(wave_profile, p.detector_oversampling)
            strip.set_data(screen_I[:, None, None] * screen_color[None, None, :])
        else:
            img = det_image()
            img = img / max(float(img.max()), 1e-12)
            strip.set_data(img[:, :, None] * screen_color[None, None, :])
        screen_hit_sc.set_offsets(np.empty((0, 2)))
        ax_p.set_xlim(0, 1.05)
        strip.set_clim(0, 1)
        set_status("wave travelling to screen..." if sim.t_step < screen_arrival_step else "")

activate()
anim = FuncAnimation(fig, update, interval=15, cache_frame_data=False)
plt.show()