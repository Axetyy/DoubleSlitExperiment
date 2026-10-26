# ui.py
from matplotlib.widgets import Slider

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