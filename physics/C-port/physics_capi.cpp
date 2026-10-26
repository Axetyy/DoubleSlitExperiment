// physics_capi.cpp - flat C interface over WaveSim2D, for ctypes / any FFI.
// physics_cpp.py builds this shared library on first use.
#include <cstdint>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>

#include "physics.h"

#if defined(_WIN32)
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

namespace
{
    thread_local std::string last_error;

    struct Handle
    {
        Params p;
        std::unique_ptr<WaveSimulator> sim;
    };
    Handle *H(void *h) { return static_cast<Handle *>(h); }

    bool valid(void *h)
    {
        if (!h)
        {
            last_error = "invalid simulator handle";
            return false;
        }
        return true;
    }

    int fail(const std::exception &e)
    {
        last_error = e.what();
        return -1;
    }
} // namespace

API const char *sim_last_error() { return last_error.c_str(); }

API void *sim_create()
{
    try
    {
        last_error.clear();
        return new Handle();
    }
    catch (const std::exception &e)
    {
        last_error = e.what();
        return nullptr;
    }
}

API void sim_destroy(void *h) { delete H(h); }

// Set a Params field by name (all values passed as double). Returns 0 ok, -1 unknown.
API int sim_set_param(void *h, const char *name, double v)
{
    last_error.clear();
    if (!valid(h) || !name || !std::isfinite(v))
    {
        if (last_error.empty())
            last_error = "parameter name and value must be valid and finite";
        return -1;
    }
    Params &p = H(h)->p;
#define SETI(n)                    \
    if (!std::strcmp(name, #n))    \
    {                              \
        if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) \
        {                            \
            last_error = "integer parameter is out of range"; \
            return -1;               \
        }                            \
        p.n = static_cast<int>(v); \
        return 0;                  \
    }
#define SETF(n)                      \
    if (!std::strcmp(name, #n))      \
    {                                \
        p.n = static_cast<float>(v); \
        return 0;                    \
    }
#define SETB(n)                 \
    if (!std::strcmp(name, #n)) \
    {                           \
        p.n = (v != 0.0);       \
        return 0;               \
    }
    SETI(x_src)
    SETI(x_wall)
    SETI(x_det)
    SETI(nx)
    SETI(ny)
    SETI(wall_thickness) SETI(sponge)
        SETI(n_slits) SETI(detector_oversampling) SETI(detector_width)
            SETF(courant) SETF(wavelength) SETF(slit_width) SETF(slit_spacing) SETF(source_angle_deg)
                SETF(source_phase) SETF(source_y_offset) SETF(wavelength_nm) SETF(slit_width_um)
                    SETF(slit_spacing_um) SETF(screen_distance_m) SETF(screen_half_height_mm) SETF(um_to_cells)
                        SETF(cells_per_m) SETF(lam_cells_min) SETF(lam_cells_max) SETF(slit_height_um)
                            SETF(slit_height)
                                SETB(coherent) SETB(particle_mode)
#undef SETI
#undef SETF
#undef SETB
    last_error = std::string("unknown parameter: ") + name;
    return -1;
}

// Construct the simulator from the params set so far (fixes nx, ny, sponge).
API int sim_build(void *h)
{
    if (!valid(h))
        return -1;
    Handle *s = H(h);
    try
    {
        s->sim = std::make_unique<WaveSimulator>(s->p);
        last_error.clear();
        return 0;
    }
    catch (const std::exception &e)
    {
        return fail(e);
    }
}

// Re-apply params (except nx/ny/sponge, which are fixed at build) and restart.
API int sim_reset(void *h)
{
    if (!valid(h))
        return -1;
    Handle *s = H(h);
    if (!s->sim)
    {
        last_error = "simulator has not been built";
        return -1;
    }
    try
    {
        Params np = s->p;
        const Params cur = s->sim->params();
        np.nx = cur.nx;
        np.ny = cur.ny;
        np.sponge = cur.sponge;
        s->sim->params() = np;
        try
        {
            s->sim->reset();
        }
        catch (...)
        {
            s->sim->params() = cur;
            throw;
        }
        last_error.clear();
        return 0;
    }
    catch (const std::exception &e)
    {
        return fail(e);
    }
}

API int sim_step_n(void *h, int n)
{
    if (!valid(h) || !H(h)->sim)
        return -1;
    try
    {
        H(h)->sim->step_n(n);
        last_error.clear();
        return 0;
    }
    catch (const std::exception &e)
    {
        return fail(e);
    }
}

API long long sim_t_step(void *h)
{
    return valid(h) && H(h)->sim ? H(h)->sim->tStep() : -1;
}

API int sim_src_y(void *h)
{
    return valid(h) && H(h)->sim ? H(h)->sim->srcY() : -1;
}

// out: ny*nx floats, row-major
API int sim_copy_field(void *h, float *out)
{
    if (!valid(h) || !H(h)->sim || !out)
    {
        last_error = "simulator and output buffer must be valid";
        return -1;
    }
    const Tensor2D &u = H(h)->sim->fieldSlice();
    for (int y = 0; y < u.rows(); ++y)
        std::memcpy(out + static_cast<size_t>(y) * u.cols(), u[y].data(), sizeof(float) * u.cols());
    return 0;
}

// out: ny floats
API int sim_copy_detector(void *h, float *out)
{
    if (!valid(h) || !H(h)->sim || !out)
    {
        last_error = "simulator and output buffer must be valid";
        return -1;
    }
    const std::vector<float> &d = H(h)->sim->detectorProfile();
    std::memcpy(out, d.data(), sizeof(float) * d.size());
    return 0;
}

// out: ny bytes (1 = slit open)
API int sim_copy_slit_open(void *h, std::uint8_t *out)
{
    if (!valid(h) || !H(h)->sim || !out)
    {
        last_error = "simulator and output buffer must be valid";
        return -1;
    }
    const auto &s = H(h)->sim->slitOpen();
    std::memcpy(out, s.data(), s.size());
    return 0;
}