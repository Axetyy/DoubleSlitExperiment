"""Matplotlib presentation for the double-slit simulation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, RadioButtons

from controller import SimulationManager
from ui.ui import (
    frame_from_snapshot,
    make_slider,
    quiet_set,
    radio_quiet,
    retarget,
)
from utils.utils import high_res_screen, runs


class MatplotlibUI:

    SURF_STEP = 4
    SURF_EVERY = 2

    def __init__(self, manager: SimulationManager):
        self.manager = manager
        self.dim = manager.dim
        self.device = manager.device
        self.state = {"paused": False, "scale": 1.0}
        self.fraunhofer_profile = np.zeros(manager.p.ny)
        self.slice_surf = None
        self.screen_surf = None
        self.wall_bars = []

        self.fig = plt.figure(figsize=(11, 8.8))
        grid = GridSpec(
            1,
            3,
            width_ratios=[4, 0.3, 1.1],
            left=0.05,
            right=0.98,
            top=0.95,
            bottom=0.40,
            wspace=0.03,
        )
        self._create_axes(grid)
        self._create_controls()
        self._connect_callbacks()

    @property
    def p(self):
        return self.manager.p

    @property
    def sim(self):
        return self.manager.sim

    @property
    def sampler(self):
        return self.manager.sampler

    @property
    def screen_rows(self):
        return self.manager.screen_rows

    def _create_axes(self, grid):
        self.ax_f = self.fig.add_subplot(grid[0])
        self.ax_s = self.fig.add_subplot(grid[1])
        self.ax_p = self.fig.add_subplot(grid[2])
        self.ax_f3 = self.fig.add_subplot(grid[0], projection="3d")
        self.ax_f3.set_visible(False)
        self.pos2 = {
            "f": self.ax_f.get_position().frozen(),
            "s": self.ax_s.get_position().frozen(),
            "p": self.ax_p.get_position().frozen(),
        }

        p = self.manager.p2
        self.im = self.ax_f.imshow(
            np.zeros((min(p.ny, p.view_ny), p.nx)),
            origin="lower",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
            aspect="auto",
        )
        self.wall_im = self.ax_f.imshow(
            np.zeros((min(p.ny, p.view_ny), p.nx, 4), np.uint8),
            origin="lower",
            interpolation="nearest",
            aspect="auto",
        )
        self.src_pts, = self.ax_f.plot([], [], "g.", ms=3)
        self.detector_line = self.ax_f.axvline(p.x_det, color="k", lw=2)
        self.hit_sc = self.ax_f.scatter(
            [], [], s=4, c="darkorange", edgecolors="none", visible=False
        )
        self.status = self.ax_f.text(
            0.02,
            0.96,
            "",
            transform=self.ax_f.transAxes,
            fontsize=9,
            bbox=dict(fc="white", ec="none", alpha=0.8),
        )
        self.ax_f.set_autoscale_on(False)

        self.status3 = self.ax_f3.text2D(
            0.02,
            0.96,
            "",
            transform=self.ax_f3.transAxes,
            fontsize=9,
            bbox=dict(fc="white", ec="none", alpha=0.8),
        )
        self.src3d, = self.ax_f3.plot([0], [0], [0], "g.", ms=8)
        self.ax_f3.set_xlabel("x", fontsize=8)
        self.ax_f3.set_ylabel("y", fontsize=8)
        self.ax_f3.set_zlabel("z", fontsize=8)
        self.ax_f3.view_init(elev=30, azim=-65)

        self.strip = self.ax_s.imshow(
            np.zeros((p.ny * p.detector_oversampling, 1, 3), dtype=np.float32),
            origin="lower",
            extent=(0, 1, -p.screen_half_height_mm, p.screen_half_height_mm),
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )
        self.screen_hit_sc = self.ax_s.scatter(
            [],
            [],
            s=10,
            c="darkorange",
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
            visible=False,
        )
        self.ax_s.set_ylabel("position (mm)", fontsize=8)

        self.line_i, = self.ax_p.plot(
            [], [], "b", lw=1.4, label="wave intensity", visible=True
        )
        self.line_h, = self.ax_p.plot(
            [],
            [],
            "k",
            lw=1,
            drawstyle="steps-mid",
            label="cumulative detections",
            visible=False,
        )
        self.line_f, = self.ax_p.plot(
            [],
            [],
            "r--",
            lw=1.2,
            label="grid Fraunhofer approximation",
            visible=True,
        )
        self.ax_p.set_autoscale_on(False)
        self.ax_p.set_xlim(0, 1)
        self.ax_p.set_title("screen profile", fontsize=9)
        self.ax_p.grid(True, alpha=0.25)
        self.ax_p.set_xlabel("normalized intensity")
        self.ax_p.set_ylabel("position (mm)")
        self.ax_p.legend(handles=[self.line_i, self.line_f], loc="upper right", fontsize=7)
        self.note = self.fig.text(0.5, 0.005, "", ha="center", fontsize=8, color="crimson")

    def _create_controls(self):
        p2 = self.manager.p2
        p3 = self.manager.p3
        rows = [0.30, 0.24, 0.18, 0.12, 0.06]
        x1, x2, width = 0.31, 0.74, 0.19
        self.s_lam = make_slider(self.fig, [x1, rows[0], width, 0.03], "wavelength (nm)", 400, 700, p2.wavelength_nm, 0.1)
        self.s_n = make_slider(self.fig, [x1, rows[1], width, 0.03], "# slits", 1, 7, p2.n_slits, 1)
        self.s_a = make_slider(self.fig, [x1, rows[2], width, 0.03], "slit width (μm)", 10, 80, p2.slit_width_um, 1)
        self.s_d = make_slider(self.fig, [x1, rows[3], width, 0.03], "slit pitch (μm)", 100, 500, p2.slit_spacing_um, 5)
        self.s_src = make_slider(self.fig, [x1, rows[4], width, 0.03], "source dist (cells)", 10, 110, p2.x_wall - p2.x_src, 1)
        self.s_l = make_slider(self.fig, [x2, rows[0], width, 0.03], "screen L (m)", 0.5, 3.5, p2.screen_distance_m, 0.1)
        self.s_ang = make_slider(self.fig, [x2, rows[1], width, 0.03], "incidence (deg)", -5, 5, p2.source_angle_deg, 0.1)
        self.s_spf = make_slider(self.fig, [x2, rows[2], width, 0.03], "steps/frame", 1, 30, 10, 1)
        self.s_ppf = make_slider(self.fig, [x2, rows[3], width, 0.03], "mean photons/frame", 1, 100, 10, 1)
        self.s_h = make_slider(self.fig, [x2, rows[4], width, 0.03], "slit height (μm) [3D]", 100, 1200, p3.slit_height_um, 10)

        self.rb_dim = RadioButtons(self.fig.add_axes([0.005, 0.27, 0.085, 0.07]), ["2D", "3D"])
        self.rb_dev = RadioButtons(self.fig.add_axes([0.005, 0.18, 0.085, 0.085]), ["CPU", "C++", "GPU"])
        self.rb_mode = RadioButtons(self.fig.add_axes([0.005, 0.11, 0.085, 0.07]), ["Wave", "Particle"])
        self.rb_coh = RadioButtons(self.fig.add_axes([0.005, 0.03, 0.085, 0.07]), ["Coherent", "Incoherent"])
        for radio in (self.rb_dim, self.rb_dev, self.rb_mode, self.rb_coh):
            for label in radio.labels:
                label.set_fontsize(8)
        self.b_reset = Button(self.fig.add_axes([0.10, 0.27, 0.07, 0.04]), "Reset")
        self.b_pause = Button(self.fig.add_axes([0.10, 0.20, 0.07, 0.04]), "Pause")

    def _connect_callbacks(self):
        for slider in (self.s_lam, self.s_n, self.s_a, self.s_d, self.s_l, self.s_ang, self.s_src, self.s_h):
            slider.on_changed(self.on_param)
        self.rb_dim.on_clicked(self.on_dim)
        self.rb_dev.on_clicked(self.on_device)
        self.rb_mode.on_clicked(self.on_mode)
        self.rb_coh.on_clicked(self.on_coherence)
        self.b_reset.on_clicked(lambda _: self.manager.reset())
        self.b_pause.on_clicked(self.on_pause)

    def set_status(self, message):
        self.status.set_text(message)
        self.status3.set_text(message)

    def set_3d_limits(self):
        xmax = self.p.x_det + 8
        self.ax_f3.set_xlim(0, xmax)
        self.ax_f3.set_ylim(0, self.p.ny)
        self.ax_f3.set_zlim(0, self.p.nz)
        self.ax_f3.set_box_aspect((xmax, self.p.ny, self.p.nz))

    def set_layout(self):
        half = self.p.screen_half_height_mm
        if self.dim == 2:
            self.ax_f.set_visible(True)
            self.ax_f3.set_visible(False)
            self.ax_f.set_position(self.pos2["f"])
            self.ax_s.set_position(self.pos2["s"])
            self.ax_p.set_position(self.pos2["p"])
            self.ax_s.set_aspect("auto", adjustable="box")
            self.strip.set_extent((0, 1, -half, half))
            self.ax_s.set_xlim(0, 1)
            self.ax_s.set_ylim(-half, half)
            self.ax_s.set_xticks([])
            self.ax_s.set_xlabel("")
            self.ax_s.set_title("detector", fontsize=8)
            self.ax_f.set_xlim(-0.5, self.p.x_det + 8.5)
            self.ax_f.set_ylim(self.screen_rows[0] - 0.5, self.screen_rows[-1] + 0.5)
            extent = (-0.5, self.p.nx - 0.5, self.screen_rows[0] - 0.5, self.screen_rows[-1] + 0.5)
            self.im.set_extent(extent)
            self.wall_im.set_extent(extent)
            self.detector_line.set_xdata([self.p.x_det, self.p.x_det])
        else:
            zh = self.manager.screen_z_half
            self.ax_f.set_visible(False)
            self.ax_f3.set_visible(True)
            self.ax_f3.set_position([0.02, 0.38, 0.52, 0.60])
            self.ax_s.set_position([0.58, 0.45, 0.22, 0.45])
            self.ax_p.set_position([0.86, 0.45, 0.12, 0.45])
            self.ax_s.set_aspect("equal", adjustable="box")
            self.strip.set_extent((-zh, zh, -half, half))
            self.ax_s.set_xlim(-zh, zh)
            self.ax_s.set_ylim(-half, half)
            self.ax_s.set_xticks([-zh, 0, zh])
            self.ax_s.set_xticklabels([f"{-zh:.1f}", "0", f"{zh:.1f}"], fontsize=7)
            self.ax_s.set_xlabel("z (mm)", fontsize=8)
            self.ax_s.set_title("detector plane", fontsize=8)
            self.set_3d_limits()
        self.ax_p.set_ylim(-half, half)
        self.ax_f.set_title(f"qualitative wave field (grid units) - {self.manager.backend_label}", fontsize=9)
        self.ax_f3.set_title(f"z = centre slice, wall, detector plane - {self.manager.backend_label}", fontsize=9)

    def refresh_static(self):
        sim = self.sim
        if self.dim == 2:
            rgba = np.zeros((len(self.screen_rows), self.p.nx, 4), np.uint8)
            rgba[~sim.slit_open[self.screen_rows], self.p.x_wall:self.p.x_wall + self.p.wall_thickness] = (25, 25, 25, 255)
            self.wall_im.set_data(rgba)
            self.src_pts.set_data(np.full(len(sim.src_y), self.p.x_src), sim.src_y)
        else:
            for artist in self.wall_bars:
                artist.remove()
            self.wall_bars = []
            zo = np.flatnonzero(sim.slit_open_z)
            z_lo, z_hi = (int(zo.min()), int(zo.max()) + 1) if zo.size else (self.p.nz // 2, self.p.nz // 2)
            blocks = []
            for y0, y1, is_open in runs(sim.slit_open):
                if not is_open:
                    blocks.append((y0, y1, 0, self.p.nz))
                else:
                    if z_lo > 0:
                        blocks.append((y0, y1, 0, z_lo))
                    if z_hi < self.p.nz:
                        blocks.append((y0, y1, z_hi, self.p.nz))
            for y0, y1, z0, z1 in blocks:
                self.wall_bars.append(self.ax_f3.bar3d(self.p.x_wall, y0, z0, self.p.wall_thickness, y1 - y0, z1 - z0, color="0.25", alpha=0.55, shade=True))
            self.src3d.set_data_3d([self.p.x_src], [int(sim.src_y[0])], [sim.src_z])
        self.fraunhofer_profile = self.manager.reference_profile()
        self.line_f.set_data(self.fraunhofer_profile, self.manager.screen_y_mm)

    def apply_mode_visuals(self):
        particle_mode = self.p.particle_mode
        self.im.set_alpha(0.3 if particle_mode else 1.0)
        self.hit_sc.set_visible(particle_mode and self.dim == 2)
        self.screen_hit_sc.set_visible(particle_mode)
        self.line_i.set_visible(not particle_mode)
        self.line_h.set_visible(False)
        self.line_f.set_label("Fraunhofer expected counts" if particle_mode else "grid Fraunhofer approximation")
        self.ax_p.set_xlabel("detections per screen bin" if particle_mode else "normalized wave intensity")
        self.ax_p.set_title("cumulative photon detections" if particle_mode else "simulated detector intensity")
        self.ax_p.legend(handles=[self.line_h, self.line_f] if particle_mode else [self.line_i, self.line_f], loc="upper right", fontsize=7)
        self.manager.sampler.reset()

    def activate(self):
        self.manager.dim = self.dim
        self.manager.p = self.manager.p3 if self.dim == 3 else self.manager.p2
        self.manager.p.coherent = self.rb_coh.value_selected == "Coherent"
        self.manager.p.particle_mode = self.rb_mode.value_selected == "Particle"
        for artist in (self.slice_surf, self.screen_surf):
            if artist is not None:
                artist.remove()
        self.slice_surf = self.screen_surf = None
        self.manager.build()
        self.configure_controls()
        self.set_layout()
        self.refresh_static()
        self.apply_mode_visuals()
        self.state["scale"] = 1.0

    def configure_controls(self):
        p = self.p
        quiet_set(self.s_lam, p.wavelength_nm)
        quiet_set(self.s_n, p.n_slits)
        quiet_set(self.s_a, p.slit_width_um)
        quiet_set(self.s_d, p.slit_spacing_um)
        quiet_set(self.s_l, p.screen_distance_m)
        quiet_set(self.s_ang, p.source_angle_deg)
        lo = 40 if self.dim == 2 else 15
        retarget(self.s_src, lo, p.x_wall - p.sponge, p.x_wall - p.x_src)
        quiet_set(self.s_h, p.slit_height_um)
        self.s_h.ax.set_visible(self.dim == 3)

    def on_param(self, _=None):
        p = self.p
        p.wavelength_nm = self.s_lam.val
        p.slit_width_um = self.s_a.val
        p.slit_spacing_um = self.s_d.val
        p.slit_height_um = self.s_h.val
        p.screen_distance_m = self.s_l.val
        p.x_src = p.x_wall - int(self.s_src.val)
        p.x_det = p.x_wall + int(round(p.cells_per_m * p.screen_distance_m))
        p.source_angle_deg = self.s_ang.val
        p.wavelength = float(np.interp(p.wavelength_nm, [400, 700], [p.lam_cells_min, p.lam_cells_max]))
        p.n_slits = int(self.s_n.val)
        p.slit_width = p.slit_width_um * p.um_to_cells
        p.slit_spacing = p.slit_spacing_um * p.um_to_cells
        p.slit_height = p.slit_height_um * p.um_to_cells
        self.manager.configure_geometry()
        if self.dim == 2:
            self.detector_line.set_xdata([p.x_det, p.x_det])
            self.ax_f.set_xlim(-0.5, p.x_det + 8.5)
        else:
            self.set_3d_limits()
        self.manager.reset()
        self.refresh_static()

    def on_dim(self, label):
        new_dim = 3 if label == "3D" else 2
        if new_dim == self.dim:
            return
        if new_dim == 3 and self.device == "GPU" and not self.manager.cuda_available:
            radio_quiet(self.rb_dim, 0)
            return
        self.dim = new_dim
        self.activate()

    def on_device(self, label):
        previous = self.device
        self.device = label
        self.manager.device = label
        try:
            self.manager.build()
        except Exception as exc:
            self.device = previous
            self.manager.device = previous
            radio_quiet(self.rb_dev, ["CPU", "C++", "GPU"].index(previous))
            self.note.set_text(f"Could not start {label}: {exc}")
            return
        self.set_layout()
        self.refresh_static()
        self.manager.sampler.reset()

    def on_mode(self, label):
        self.p.particle_mode = label == "Particle"
        self.apply_mode_visuals()

    def on_coherence(self, label):
        self.p.coherent = label == "Coherent"
        self.manager.reset()
        self.refresh_static()

    def on_pause(self, _):
        self.state["paused"] = not self.state["paused"]
        self.b_pause.label.set_text("Resume" if self.state["paused"] else "Pause")

    def draw_3d(self, display):
        cmap = plt.get_cmap("coolwarm")
        xs = min(self.p.nx, self.p.x_det + 9)
        sub = display[::self.SURF_STEP, :xs:self.SURF_STEP]
        rgba = cmap(np.clip(0.5 + 0.5 * sub / max(self.state["scale"], 1e-6), 0, 1))
        xi = np.arange(0, xs, self.SURF_STEP)[:sub.shape[1]]
        yi = np.arange(0, self.p.ny, self.SURF_STEP)[:sub.shape[0]]
        x_grid, y_grid = np.meshgrid(xi, yi)
        if self.slice_surf is not None:
            self.slice_surf.remove()
        self.slice_surf = self.ax_f3.plot_surface(x_grid, y_grid, np.full(x_grid.shape, self.p.nz / 2.0), facecolors=rgba, rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)

        if self.p.particle_mode:
            hist = self.sampler.hist
            colors = plt.get_cmap("inferno")(hist / max(float(hist.max()), 1.0))
        else:
            image = self.manager.detector_image()
            image = image / max(float(image.max()), 1e-12)
            colors = np.ones(image.shape + (4,))
            colors[..., :3] = image[..., None] * self.manager.screen_color[None, None, :]
        colors = colors[::self.SURF_STEP, ::self.SURF_STEP]
        yi = np.arange(0, self.p.ny, self.SURF_STEP)[:colors.shape[0]]
        zi = np.arange(0, self.p.nz, self.SURF_STEP)[:colors.shape[1]]
        y_grid, z_grid = np.meshgrid(yi, zi, indexing="ij")
        if self.screen_surf is not None:
            self.screen_surf.remove()
        self.screen_surf = self.ax_f3.plot_surface(np.full(y_grid.shape, float(self.p.x_det)), y_grid, z_grid, facecolors=colors, rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)

    def update(self, frame_number):
        if not self.state["paused"]:
            self.manager.step(int(self.s_spf.val), int(self.s_ppf.val))
        data = frame_from_snapshot(self.manager.snapshot())
        display = np.sign(data.field) * np.sqrt(np.abs(data.field))
        if frame_number % 5 == 0:
            region = display[::2, self.p.x_src + 20::2]
            if region.size:
                target = max(float(np.percentile(np.abs(region), 99.5)), 0.02)
                self.state["scale"] = 0.8 * self.state["scale"] + 0.2 * target
        if self.dim == 2:
            self.im.set_data(display[data.screen_rows])
            self.im.set_clim(-self.state["scale"], self.state["scale"])
        elif frame_number % self.SURF_EVERY == 0:
            self.draw_3d(display)

        arrived = data.time_step >= data.arrival_step
        half = self.p.screen_half_height_mm
        if data.particle_mode:
            hist = data.particle_hist
            counts = hist[data.screen_rows] if self.dim == 2 else hist.sum(axis=1)
            max_count = float(counts.max())
            total_count = float(counts.sum())
            self.line_h.set_data(counts, data.screen_y_mm)
            self.line_h.set_visible(max_count > 0)
            expected = data.reference_profile * total_count / max(data.reference_profile.sum(), 1e-12)
            self.line_f.set_data(expected, data.screen_y_mm)
            self.ax_p.set_xlim(0, max(1, max(max_count, float(expected.max())) * 1.1))
            if self.dim == 2:
                self.hit_sc.set_offsets(data.recent_hits)
                offsets = np.column_stack((np.full(len(data.recent_hits), 0.5), (data.recent_hits[:, 1] - data.screen_rows[0] + 0.5) * self.manager.screen_pixel_size_mm - half)) if len(data.recent_hits) else np.empty((0, 2))
                self.screen_hit_sc.set_offsets(offsets)
                self.strip.set_data(high_res_screen(counts, self.p.detector_oversampling)[:, None])
                self.strip.set_clim(0, max(1, max_count))
            else:
                offsets = np.column_stack(((data.recent_hits[:, 1] + 0.5) * self.manager.screen_pixel_size_mm - self.manager.screen_z_half, (data.recent_hits[:, 0] + 0.5) * self.manager.screen_pixel_size_mm - half)) if len(data.recent_hits) else np.empty((0, 2))
                self.screen_hit_sc.set_offsets(offsets)
                self.strip.set_data(hist)
                self.strip.set_clim(0, max(1.0, float(hist.max())))
            self.set_status("wave travelling to screen..." if not arrived else (f"{int(total_count)} detections" if total_count else "wave reached screen - accumulating detections..."))
        else:
            profile = data.detector_profile[data.screen_rows] if self.dim == 2 else data.detector_profile
            peak = float(profile.max())
            wave_profile = profile / max(peak, 1e-12)
            self.line_i.set_data(wave_profile, data.screen_y_mm)
            self.line_i.set_visible(arrived and peak > 1e-12)
            if self.dim == 2:
                self.strip.set_data(high_res_screen(wave_profile, self.p.detector_oversampling)[:, None, None] * data.screen_color[None, None, :])
            else:
                image = data.detector_image / max(float(data.detector_image.max()), 1e-12)
                self.strip.set_data(image[:, :, None] * data.screen_color[None, None, :])
            self.screen_hit_sc.set_offsets(np.empty((0, 2)))
            self.ax_p.set_xlim(0, 1.05)
            self.strip.set_clim(0, 1)
            self.set_status("wave travelling to screen..." if not arrived else "")

    def show(self):
        self.activate()
        self.animation = FuncAnimation(self.fig, self.update, interval=15, cache_frame_data=True)
        plt.show()
