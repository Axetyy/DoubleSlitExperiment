##  Project

This the repo for the assignment, I've written the formulas I implemented below in a nice form (courtesy of Claudius Maximus Pro)

### Code map

- `main.py` - main UI
- `physics/core.py` - unified implementation for the Numba, PyTorch, and C++
  2D/3D wave backends.
- `physics/` module files - compatibility imports for existing backend paths.
- `vistaengine.py` - 3D UI.
- `sampling.py` and `sampling3d.py` - particle/detection sampling.
- `reference.py` - Fraunhofer reference patterns for compariso.
- `config.py` - simulation parameters.

The physics backends are alternative engines or dimensions used by the same
simulation UI, not separate simulations to launch individually. Import them
from `physics` for the unified API; existing backend-specific imports remain
supported.

### Tests

Run the physics and core-helper tests with `python tests.py`. To measure line
coverage for those modules, install `coverage` and run:

```text
python -m coverage run tests.py
python -m coverage report --include="physics/core.py,config.py,reference.py,sampling/sampling.py,sampling/sampling3d.py,utils.py"
```

## Tech stack

But first, the tech I used to do this: 
- Python
  -  Numpy
  -  numba ( for parallelization and caching )
  -  matplotlib ( for the UI , nothing fancy)
  -  pyvista, vtk
- C++ 


## Formulas

### Source formula
$$
u_s(t)=A(t)\sin(\omega t+\phi_0),
$$

- $A(t)$ is the source amplitude with respect to time
- $\omega$ is the angular frequency
- $\phi_0$ is the initial phase.

The angular frequency and wavenumber are related to the wavelength by

$$
k=\frac{2\pi}{\lambda},
\qquad
\omega=ck,
$$

### Wave Propagation

Scalar wave equation ( not the probabilistic one ) 

$$
\frac{\partial^2 u}{\partial t^2}=c^2\nabla^2u+S(x,y,t),
$$


$$
\nabla^2u=
\frac{\partial^2u}{\partial x^2}
+\frac{\partial^2u}{\partial y^2}.
$$

For a grid ( which is just a matrix ), we use: 

$$
L_{i,j}^{n}=
u_{i+1,j}^{n}+u_{i-1,j}^{n}
+u_{i,j+1}^{n}+u_{i,j-1}^{n}
-4u_{i,j}^{n}.
$$
> Also called the discrete laplacian 


$$
u_{i,j}^{n+1}
=2u_{i,j}^{n}-u_{i,j}^{n-1}
+C^2L_{i,j}^{n},
$$

where

$$
C=\frac{c\Delta t}{\Delta}.
$$

is the Courant number.
[See more about Courant](https://en.wikipedia.org/wiki/Richard_Courant)

### Boundaries

To reduce artificial reflections from the edges of the computational domain, we basically place a big sponge on the sides and in the field itself to simulate
the slight dampening of waves.

$$
\frac{\partial^2u}{\partial t^2}
+\gamma(x,y)\frac{\partial u}{\partial t}
=c^2\nabla^2u+S(x,y,t).
$$

The damping coefficient $\gamma(x,y)$ is small in the central region and increases near the boundaries. This approximates an absorbing layer that reduces the amplitude of outgoing waves before they reach the grid edges.

The corresponding discrete update is

$$
\begin{aligned}
u_{i,j}^{n+1}={}&
\frac{2u_{i,j}^{n}+C^2L_{i,j}^{n}}
{1+\gamma_{i,j}\Delta t/2}\\
&-\frac{1-\gamma_{i,j}\Delta t/2}
{1+\gamma_{i,j}\Delta t/2}u_{i,j}^{n-1}.
\end{aligned}
$$

### The Double-Slits

It's just a mask

$$
M(x,y)=
\begin{cases}
1,&\text{in free space and inside the slits},\\
0,&\text{inside the opaque barrier}.
\end{cases}
$$


The slits are not treated as independent artificial sources. Their outgoing waves emerge from the numerical propagation of the incident field through the openings.

> Really, the wall is just a mask where the wave cannot propagate forward.

### Superposition and Interference

Because the scalar wave equation is linear, the total field is the sum of the contributions from the two slits:

$$
u(x,y,t)=u_1(x,y,t)+u_2(x,y,t).
$$

For two harmonic waves with equal amplitude $A$ and phase difference $\Delta\phi$, their sum is

$$
u=
2A\cos\left(\frac{\Delta\phi}{2}\right)
\cos\left(\omega t+\frac{\phi_1+\phi_2}{2}\right).
$$

The resulting amplitude depends on the phase difference:

* **Constructive interference:** $\Delta\phi=2m\pi$.
* **Destructive interference:** $\Delta\phi=(2m+1)\pi$.

For two slits separated by a distance $d$, the far-field path difference is approximately

$$
\Delta r=d\sin\theta.
$$

The phase difference is therefore

$$
\Delta\phi=\frac{2\pi d\sin\theta}{\lambda}.
$$

The approximate bright-fringe condition is

$$
d\sin\theta=m\lambda,
$$

and the dark-fringe condition is

$$
d\sin\theta=\left(m+\frac12\right)\lambda,
$$

where $m$ is an integer.

### Intensity

The detector measures a quantity proportional to the time-averaged squared wave field (Just so it's normalized and you could see the cool patterns) : 

$$
I(x,y)\propto\left\langle u(x,y,t)^2\right\rangle_t.
$$

For two contributions,

$$
\begin{aligned}
I&\propto\left\langle(u_1+u_2)^2\right\rangle\\
&=\langle u_1^2\rangle+\langle u_2^2\rangle
+2\langle u_1u_2\rangle.
\end{aligned}
$$

The cross term $2\langle u_1u_2\rangle$ produces interference.

The simulation averages the squared field over time and across a finite number of detector columns ( See `config.py` ). It uses an exponential running average:

$$
I_i^{n+1}
=I_i^n+
\frac{\overline{u_i^2}^{\,n+1}-I_i^n}{N},
$$


### Coherence

The simulation can also add a bit of realism ( i.e noise to the patterns) by making the waves incoherent.
So we change the phase to be

$$
\phi_n=\omega\Delta t\,n+\phi_0+\eta_n,
$$

and

$$
\eta_n=\eta_{n-1}+\sigma\xi_n,
\qquad
\xi_n\sim\mathcal{N}(0,1).
$$


These fluctuations model a source with a changing phase. 

### Fraunhofer Diffraction Reference

Very cool stuff overhere.

[See more about Fraunhofer](https://en.wikipedia.org/wiki/Joseph_von_Fraunhofer)
He literally invented the spectroscope and even put the cool fringes in a formula. 

In the far-field regime, the complex field at observation angle $\theta$ is proportional to the Fourier transform of the field across the aperture:

$$
U(\theta)\propto
\int_{\mathrm{aperture}}
U_{\mathrm{ap}}(y')
e^{-iky'\sin\theta}\,dy'.
$$

The corresponding intensity is

$$
I(\theta)\propto|U(\theta)|^2.
$$

For a point source at a finite distance, the incident aperture field can be approximated by

$$
U_{\mathrm{ap}}(y')=
\frac{e^{ikr(y')}}{\sqrt{r(y')}},
$$

where

$$
r(y')=\sqrt{L_s^2+(y'-y_s)^2}.
$$
