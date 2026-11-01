# Double Slit Experiment

**A numerical simulation of wave propagation, diffraction, and interference through a double slit.**

A computational exploration of the classical wave equation, implemented using multiple numerical backends and extended with particle detection, coherence effects, and analytical Fraunhofer references.

<!-- Replace these paths with your actual screenshots -->
<p align="center">
  <img src="assets/screenshots/simulation-2d.png" width="80%" alt="2D double slit wave simulation">
</p>

<p align="center">
  <img src="assets/screenshots/simulation-3d.png" width="80%" alt="3D double slit wave simulation">
</p>
---

## Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Architecture](#project-architecture)
- [Theoretical Background](#theoretical-background)
  - [Source and Harmonic Waves](#1-source-and-harmonic-waves)
  - [Wave Propagation](#2-wave-propagation)
  - [Numerical Discretization](#3-numerical-discretization)
  - [Absorbing Boundaries](#4-absorbing-boundaries)
  - [The Double-Slit Geometry](#5-the-double-slit-geometry)
  - [Superposition and Interference](#6-superposition-and-interference)
  - [Intensity and Detection](#7-intensity-and-detection)
  - [Coherence](#8-coherence)
  - [Fraunhofer Diffraction](#9-fraunhofer-diffraction)
- [Installation](#installation)
- [Running the Simulation](#running-the-simulation)
- [Tests](#tests)
- [Documentation](#documentation)

---

## Overview

The project numerically solves the scalar wave equation to reproduce the propagation of waves through a double-slit barrier.

Rather than treating the two slits as independent sources, the simulation propagates an incident wave through a spatial mask. The resulting interference pattern emerges naturally from the superposition of the propagated fields.

The project also explores the effects of:
- Different numerical implementations and computational backends.
- Two-dimensional and three-dimensional propagation.
- Finite source coherence.
- Particle and detector sampling.
- Absorbing boundary conditions.
- Comparison with analytical Fraunhofer diffraction patterns.

## Features

- **Multiple computational backends:** Numba, PyTorch, and C++.
- **2D and 3D simulations:** Explore wave propagation in different spatial dimensions.
- **Double-slit interference:** Observe constructive and destructive interference.
- **Coherence control:** Introduce phase fluctuations to study their effects on interference.
- **Particle detection:** Sample detection events from the resulting intensity distribution.
- **Analytical reference:** Compare numerical results against Fraunhofer diffraction.
- **Interactive visualization:** Explore the simulation through the graphical interface.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main application and simulation logic |
| NumPy | Numerical operations and array manipulation |
| Numba | JIT compilation, parallelization, and caching |
| PyTorch | Alternative computational backend |
| Matplotlib | Visualization and graphical interface |
| PyVista | 3D visualization |
| VTK | 3D rendering infrastructure |
| C++ | Native computational backend |

## Project Architecture

The simulation uses a unified physics interface. The available backends and spatial dimensions are alternative implementations of the same simulation rather than independent applications.

```text
DoubleSlitExperiment/
│
├── main.py                 # Main application and UI
├── config.py               # Simulation parameters
├── reference.py            # Fraunhofer reference patterns
├── utils.py                # Shared helper functions
├── vistaengine.py          # 3D visualization
│
├── physics/
│   ├── core.py             # Unified physics implementation
│   └── ...                 # Backend compatibility modules
│
├── sampling/
│   ├── sampling.py         # 2D particle/detection sampling
│   └── sampling3d.py       # 3D particle/detection sampling
│
├── tests.py                # Physics and helper tests
│
├── assets/
│   
└── README.md
```

The physics backends should be imported through the unified `physics` API. Existing backend-specific imports remain supported for compatibility.

---

# Theoretical Background

The simulation is based on classical scalar wave propagation. The following sections describe the mathematical model and numerical approximations used in the implementation.

## 1. Source and Harmonic Waves

The source generates a harmonic wave described by:

$$
u_s(t)=A(t)\sin(\omega t+\phi_0)
$$

where:

- $A(t)$ is the source amplitude.
- $\omega$ is the angular frequency.
- $\phi_0$ is the initial phase.

Angular frequency and wavenumber are related to the wavelength by:

$$
k=\frac{2\pi}{\lambda},
\qquad
\omega=ck
$$

where $c$ is the wave propagation speed.

## 2. Wave Propagation

The simulation solves the scalar wave equation:

$$
\boxed{
\frac{\partial^2u}{\partial t^2}
=
c^2\nabla^2u+S(x,y,t)
}
$$

Here, $u$ represents the scalar wave field and $S$ represents the source term.

For two spatial dimensions, the Laplacian is:

$$
\nabla^2u=
\frac{\partial^2u}{\partial x^2}
+
\frac{\partial^2u}{\partial y^2}
$$

This is a classical wave equation, not a probabilistic wavefunction evolution equation.

## 3. Numerical Discretization

To solve the equation computationally, the continuous spatial domain is represented by a discrete grid.

Using a square grid with spacing $\Delta$, the discrete Laplacian becomes:

$$
L_{i,j}^{n}=
u_{i+1,j}^{n}+u_{i-1,j}^{n}
+u_{i,j+1}^{n}+u_{i,j-1}^{n}
-4u_{i,j}^{n}
$$

This is the five-point discrete Laplacian stencil.

The corresponding explicit time-stepping scheme is:

$$
\boxed{
u_{i,j}^{n+1}
=
2u_{i,j}^{n}
-u_{i,j}^{n-1}
+C^2L_{i,j}^{n}
}
$$

where the Courant number is:

$$
C=\frac{c\Delta t}{\Delta}
$$

For the standard two-dimensional scheme on a square grid, the stability condition is:

$$
C\leq\frac{1}{\sqrt{2}}
$$

The stability restriction is important because choosing a time step that is too large can introduce numerical instabilities.

[Further reading: Courant–Friedrichs–Lewy condition](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition)

## 4. Absorbing Boundaries

A finite computational grid introduces an artificial problem: waves reaching the edges can reflect back into the simulation.

To reduce these unwanted reflections, a damping layer is introduced near the boundaries.

The modified wave equation is:

$$
\boxed{
\frac{\partial^2u}{\partial t^2}
+
\gamma(x,y)\frac{\partial u}{\partial t}
=
c^2\nabla^2u+S(x,y,t)
}
$$

The damping coefficient $\gamma(x,y)$ is small in the central region and increases towards the boundaries.

This acts as a numerical sponge, reducing the amplitude of outgoing waves before they reach the grid edges.

Using central finite differences for the temporal derivatives, the discrete update becomes:

$$
\begin{aligned}
u_{i,j}^{n+1}={}&
\frac{2u_{i,j}^{n}+C^2L_{i,j}^{n}
+\Delta t^2S_{i,j}^{n}}
{1+\gamma_{i,j}\Delta t/2}\\
&-
\frac{1-\gamma_{i,j}\Delta t/2}
{1+\gamma_{i,j}\Delta t/2}
u_{i,j}^{n-1}
\end{aligned}
$$

The source term is included here when it is incorporated directly into this update.

## 5. The Double-Slit Geometry

The double-slit barrier is represented by a spatial mask:

$$
M(x,y)=
\begin{cases}
1,&\text{in free space and inside the slits}\\
0,&\text{inside the opaque barrier}
\end{cases}
$$

The mask determines where the wave is allowed to propagate.

The slits are not treated as independent artificial sources. Their outgoing waves emerge naturally from the numerical propagation of the incident field through the openings.

In other words, the wall is simply a region where the wave cannot propagate forward.

## 6. Superposition and Interference

Because the scalar wave equation is linear, the total field can be expressed as the sum of the contributions from both slits:

$$
u(x,y,t)=u_1(x,y,t)+u_2(x,y,t)
$$

For two harmonic waves with equal amplitude $A$ and phase difference $\Delta\phi$, their sum is:

$$
u=
2A\cos\left(\frac{\Delta\phi}{2}\right)
\cos\left(
\omega t+\frac{\phi_1+\phi_2}{2}
\right)
$$

The resulting amplitude depends on the phase difference.

**Constructive interference:**

$$
\Delta\phi=2m\pi
$$

**Destructive interference:**

$$
\Delta\phi=(2m+1)\pi
$$

For two slits separated by distance $d$, the far-field path difference is approximately:

$$
\Delta r=d\sin\theta
$$

The corresponding phase difference is:

$$
\boxed{
\Delta\phi=
\frac{2\pi d\sin\theta}{\lambda}
}
$$

The approximate bright-fringe condition is:

$$
d\sin\theta=m\lambda
$$

while the dark-fringe condition is:

$$
d\sin\theta=
\left(m+\frac12\right)\lambda
$$

where $m$ is an integer.

## 7. Intensity and Detection

The detector measures a quantity proportional to the time-averaged squared wave field:

$$
\boxed{
I(x,y)\propto
\left\langle u(x,y,t)^2\right\rangle_t
}
$$

For two contributions:

$$
\begin{aligned}
I&\propto\left\langle(u_1+u_2)^2\right\rangle\\
&=\langle u_1^2\rangle+\langle u_2^2\rangle
+2\langle u_1u_2\rangle
\end{aligned}
$$

The cross term,

$$
2\langle u_1u_2\rangle
$$

is responsible for the interference pattern.

The simulation estimates the intensity using an exponential running average:

$$
I_i^{n+1}
=
I_i^n+
\frac{\overline{u_i^2}^{\,n+1}-I_i^n}{N}
$$

where $N$ controls the averaging rate.

A larger $N$ results in slower updates and stronger temporal smoothing.

The detector also uses a finite number of spatial columns, as configured in `config.py`.

## 8. Coherence

Real sources are not necessarily perfectly coherent. Their phase can fluctuate over time, reducing the visibility of the interference pattern.

To model this, the simulation introduces a stochastic phase:

$$
\phi_n=
\omega\Delta t\,n+\phi_0+\eta_n
$$

where the phase fluctuation follows a random walk:

$$
\eta_n=\eta_{n-1}+\sigma\xi_n
$$

with:

$$
\xi_n\sim\mathcal{N}(0,1)
$$

Here, $\sigma$ controls the magnitude of the phase fluctuations.

These fluctuations model phase diffusion: the source gradually loses phase stability, which can reduce interference contrast.

## 9. Fraunhofer Diffraction

The numerical simulation can be compared against an analytical far-field reference using Fraunhofer diffraction.

In the far-field regime, the complex field at observation angle $\theta$ is proportional to the Fourier transform of the field across the aperture:

$$
\boxed{
U(\theta)\propto
\int_{\mathrm{aperture}}
U_{\mathrm{ap}}(y')
e^{-iky'\sin\theta}\,dy'
}
$$

The corresponding intensity is:

$$
I(\theta)\propto|U(\theta)|^2
$$

For a point source at a finite distance, the incident aperture field can be approximated by:

$$
U_{\mathrm{ap}}(y')=
\frac{e^{ikr(y')}}{\sqrt{r(y')}}
$$

where:

$$
r(y')=
\sqrt{L_s^2+(y'-y_s)^2}
$$

This amplitude scaling corresponds to a two-dimensional cylindrical-wave model.

The analytical reference provides a useful way to compare the numerical interference pattern against the expected far-field behaviour.

[Further reading: Fraunhofer diffraction](https://en.wikipedia.org/wiki/Fraunhofer_diffraction)

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Axetyy/DoubleSlitExperiment.git
cd DoubleSlitExperiment
```

Install the required Python dependencies:

```bash
pip install numpy numba matplotlib pyvista vtk torch
```

Additional requirements may depend on the selected computational backend.

# Running the Simulation

Run the main application:

```bash
python main.py
```

The available computational backends and simulation dimensions can be selected through the application.

# Tests

Run the physics and core-helper tests:

```bash
python tests.py
```

To measure line coverage:

```bash
pip install coverage

python -m coverage run tests.py

python -m coverage report --include="physics/core.py,config.py,reference.py,sampling/sampling.py,sampling/sampling3d.py,utils.py"
```

# Documentation

The project API documentation is generated using Doxygen.

To generate the HTML documentation:

```bash
doxygen Doxyfile
```

The generated documentation can be opened locally through:

```text
docs/generated/html/index.html
```

---
