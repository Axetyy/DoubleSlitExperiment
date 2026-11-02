# ui.py
from matplotlib.widgets import Slider
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WaveFrame:
    """Format-neutral data passed from a simulation to a renderer."""

    dimension: int
    field: np.ndarray
    detector_profile: np.ndarray
    detector_image: Any
    reference_profile: np.ndarray
    screen_rows: np.ndarray
    screen_y_mm: np.ndarray
    screen_color: Any
    particle_mode: bool
    particle_hist: np.ndarray
    recent_hits: np.ndarray
    time_step: int
    arrival_step: int


def frame_from_snapshot(snapshot):
    """Convert a manager snapshot into a renderer-friendly value object."""
    return WaveFrame(**snapshot)

def make_slider(fig, rect, label, lo, hi, init, step):
    return Slider(fig.add_axes(rect), label, lo, hi, valinit=init, valstep=step)

def quiet_set(slider, val):
    ev = slider.eventson
    slider.eventson = False
    try:
        slider.set_val(val)
    finally:
        slider.eventson = ev

def retarget(slider, lo, hi, val):
    ev = slider.eventson
    slider.eventson = False
    try:
        slider.valmin, slider.valmax = lo, hi
        slider.ax.set_xlim(lo, hi)
        slider.set_val(val)
    finally:
        slider.eventson = ev

def radio_quiet(rb, index):
    ev = rb.eventson
    rb.eventson = False
    try:
        rb.set_active(index)
    finally:
        rb.eventson = ev


def say(msg,note:any)->None:
    """
    Display a message in the note area.
    """
    note.set_text(msg)