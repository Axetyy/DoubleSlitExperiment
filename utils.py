# utils.py
import numpy as np

def wavelength_to_rgb(wavelength_nm):
    w = np.asarray(wavelength_nm, dtype=float)
    r = np.zeros_like(w)
    g = np.zeros_like(w)
    b = np.zeros_like(w)
 
    m = (w >= 380) & (w < 440)
    r[m] = -(w[m] - 440) / (440 - 380)
    g[m] = 0
    b[m] = 1
 
    m = (w >= 440) & (w < 490)
    r[m] = 0
    g[m] = (w[m] - 440) / (490 - 440)
    b[m] = 1
 
    m = (w >= 490) & (w < 510)
    r[m] = 0
    g[m] = 1
    b[m] = -(w[m] - 510) / (510 - 490)
 
    m = (w >= 510) & (w < 580)
    r[m] = (w[m] - 510) / (580 - 510)
    g[m] = 1
    b[m] = 0
 
    m = (w >= 580) & (w < 645)
    r[m] = 1
    g[m] = -(w[m] - 645) / (645 - 580)
    b[m] = 0
 
    m = (w >= 645) & (w <= 750)
    r[m] = 1
    g[m] = 0
    b[m] = 0
 
    factor = np.ones_like(w)
    m = (w >= 380) & (w < 420)
    factor[m] = 0.3 + 0.7 * ((w[m] - 380) / 40)
 
    m = (w > 700) & (w <= 750)
    factor[m] = 0.3 + 0.7 * ((750 - w[m]) / 50)
 
    return np.stack([r * factor, g * factor, b * factor], axis=-1)

def high_res_screen(intensity, oversampling=4):
    ny = len(intensity)
    y = np.arange(ny)
    y_hi = (np.arange(ny * oversampling) + 0.5) / oversampling - 0.5
    return np.interp(y_hi, y, intensity)

def runs(mask):
    out, start = [], 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            out.append((start, i, bool(mask[start])))
            start = i
    return out