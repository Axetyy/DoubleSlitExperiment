#include <vector>
#include "physics.h"
#include <algorithm>
#include <cmath>

namespace math_stuff
{
    constexpr double kPi = 3.14159265358979323846;

    double deg2rad(double d) { return d * kPi / 180.0; }
    // Interpolation
    double interp(double x, double x0, double x1, double y0, double y1)
    {
        if (x <= x0)
            return y0;
        if (x >= x1)
            return y1;
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    }
}

namespace
{
void validateParams(const Params &p)
{
    if (p.nx <= 0 || p.ny <= 0 || p.sponge <= 0 ||
        p.sponge > (p.ny - 1) / 2 || p.sponge > (p.nx - 1) / 2)
        throw std::invalid_argument("grid dimensions and sponge width are invalid");
    if (p.x_src < 0 || p.x_src >= p.nx ||
        p.x_wall < 0 || p.x_wall > p.nx ||
        p.x_det < 0 || p.x_det >= p.nx ||
        p.wall_thickness < 0 || p.detector_width <= 0 ||
        p.n_slits <= 0)
        throw std::invalid_argument("source, wall, detector, or slit parameters are invalid");
    if (!std::isfinite(p.courant) || p.courant <= 0.0f ||
        !std::isfinite(p.wavelength) || p.wavelength <= 0.0f ||
        !std::isfinite(p.slit_width) || p.slit_width < 0.0f ||
        !std::isfinite(p.slit_spacing) || p.slit_spacing < 0.0f ||
        !std::isfinite(p.source_angle_deg))
        throw std::invalid_argument("wave parameters must be finite and in range");
}
}

void Tensor2D::fill(float v)
{
    for (auto &r : d_)
        std::fill(r.begin(), r.end(), v);
}
void Tensor2D::assign(const Tensor2D &other)
{
    requireSameShape(other, "assign");
    for (int y = 0; y < rows_; ++y)
    {
        std::copy(other.d_[y].begin(), other.d_[y].end(), d_[y].begin());
    }
}
Tensor2D &Tensor2D::operator+=(const Tensor2D &o)
{
    requireSameShape(o, "+=");
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < rows_; ++y)
    {
        float *a = d_[y].data();
        const float *b = o.d_[y].data();
        for (int x = 0; x < cols_; ++x)
            a[x] += b[x];
    }
    return *this;
}

Tensor2D &Tensor2D::operator-=(const Tensor2D &o)
{
    requireSameShape(o, "-=");
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < rows_; ++y)
    {
        float *a = d_[y].data();
        const float *b = o.d_[y].data();
        for (int x = 0; x < cols_; ++x)
            a[x] -= b[x];
    }
    return *this;
}

Tensor2D &Tensor2D::operator*=(const Tensor2D &o)
{
    requireSameShape(o, "*=");
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < rows_; ++y)
    {
        float *a = d_[y].data();
        const float *b = o.d_[y].data();
        for (int x = 0; x < cols_; ++x)
            a[x] *= b[x];
    }
    return *this;
}

Tensor2D &Tensor2D::operator*=(float s)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < rows_; ++y)
    {
        float *a = d_[y].data();
        for (int x = 0; x < cols_; ++x)
            a[x] *= s;
    }
    return *this;
}

Tensor2D &Tensor2D::operator+=(float s)
{
    for (auto &r : d_)
    {
        for (float &v : r)
            v += s;
    }
    return *this;
}

Tensor2D &Tensor2D::addcmul(const Tensor2D &a, const Tensor2D &b, float alpha)
{
    requireSameShape(a, "addcmul");
    requireSameShape(b, "addcmul");
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < rows_; ++y)
    {
        float *t = d_[y].data();
        const float *pa = a.d_[y].data();
        const float *pb = b.d_[y].data();
        for (int x = 0; x < cols_; ++x)
            t[x] += alpha * pa[x] * pb[x];
    }
    return *this;
}
Tensor2D Tensor2D::matmul(const Tensor2D &o) const
{
    if (cols_ != o.rows_)
        throw std::invalid_argument("matmul: inner dimensions do not match");
    Tensor2D out(rows_, o.cols_, 0.0f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < rows_; ++i)
    {
        float *c = out.d_[i].data();
        for (int k = 0; k < cols_; ++k)
        {
            const float a = d_[i][k];
            const float *b = o.d_[k].data();
            for (int j = 0; j < o.cols_; ++j)
                c[j] += a * b[j];
        }
    }
    return out;
}

void Tensor2D::laplacianInto(Tensor2D &out) const
{
    requireSameShape(out, "laplacianInto");
    if (rows_ == 0 || cols_ == 0)
        return;

    out.d_[0].assign(cols_, 0.0f);
    out.d_[rows_ - 1].assign(cols_, 0.0f);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 1; y < rows_ - 1; ++y)
    {
        const float *c = d_[y].data();
        const float *up = d_[y - 1].data();
        const float *dn = d_[y + 1].data();
        float *o = out.d_[y].data();
        o[0] = 0.0f;
        o[cols_ - 1] = 0.0f;
        for (int x = 1; x < cols_ - 1; ++x)
            o[x] = up[x] + dn[x] + c[x - 1] + c[x + 1] - 4.0f * c[x];
    }
}

void Tensor2D::rowMeanSq(int x0, int x1, std::vector<float> &out) const
{
    x0 = std::max(0, x0);
    x1 = std::min(cols_ - 1, x1);
    if (x0 > x1)
        throw std::invalid_argument("rowMeanSq: empty column range");
    out.assign(rows_, 0.0f);
    const float inv = 1.0f / static_cast<float>(x1 - x0 + 1);
    for (int y = 0; y < rows_; ++y)
    {
        const float *r = d_[y].data();
        float s = 0.0f;
        for (int x = x0; x <= x1; ++x)
            s += r[x] * r[x];
        out[y] = s * inv;
    }
}

std::vector<float> Tensor2D::flatten() const
{
    std::vector<float> f;
    f.reserve(static_cast<size_t>(rows_) * cols_);
    for (const auto &r : d_)
        f.insert(f.end(), r.begin(), r.end());
    return f;
}

WaveSimulator::WaveSimulator(const Params &p, std::uint64_t seed) : p_(p)
{
    validateParams(p_);

    if (seed == 0)
    {
        std::random_device rd;
        seed = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
    }
    rng_.seed(seed);

    u_ = Tensor2D(p_.ny, p_.nx);
    u_next_ = Tensor2D(p_.ny, p_.nx);
    u_prev_ = Tensor2D(p_.ny, p_.nx);
    lap_ = Tensor2D(p_.ny, p_.nx);
    I_det_.assign(p_.ny, 0.0f);
    buildAbsorber();
    rebuild(true);
}
void WaveSimulator::buildAbsorber()
{
    const int nx = p_.nx, ny = p_.ny;
    const double sp = p_.sponge;

    std::vector<float> depth_x(nx), depth_y(ny);
    for (int x = 0; x < nx; ++x)
    {
        depth_x[x] = static_cast<float>(
            std::max(0.0, sp - std::min(x, nx - 1 - x)) / sp);
    }
    for (int y = 0; y < ny; ++y)
    {
        depth_y[y] = static_cast<float>(
            std::max(0.0, sp - std::min(y, ny - 1 - y)) / sp);
    }

    dampening_prev_ = Tensor2D(ny, nx);
    dampening_next_ = Tensor2D(ny, nx);

    for (int y = 0; y < ny; ++y)
    {
        for (int x = 0; x < nx; ++x)
        {
            const float e = std::max(depth_y[y], depth_x[x]);
            const float s = 0.6f * e * e;
            dampening_prev_(y, x) = 1.0f - 0.5f * s;
            dampening_next_(y, x) = 1.0f / (1.0f + 0.5f * s);
        }
    }
}

void WaveSimulator::rebuild(bool fresh)
{
    const int nx = p_.nx, ny = p_.ny;
    const double yc = ny / 2.0;

    // Slit centres / open rows
    slit_centers_.resize(p_.n_slits);
    for (int i = 0; i < p_.n_slits; ++i)
        slit_centers_[i] = yc + (i - (p_.n_slits - 1) / 2.0) * p_.slit_spacing;

    slit_open_.assign(ny, 0);
    for (int y = 0; y < ny; ++y)
        for (double c : slit_centers_)
            if (std::fabs(y - c) <= p_.slit_width / 2.0)
            {
                slit_open_[y] = 1;
                break;
            }

    // Wall mask: 1 = open, 0 = solid
    wall_mask_ = Tensor2D(ny, nx, 1.0f);
    const int xw0 = std::max(0, p_.x_wall);
    const int xw1 = p_.x_wall + std::min(p_.wall_thickness, nx - p_.x_wall);
    for (int y = 0; y < ny; ++y)
        for (int x = xw0; x < xw1; ++x)
            wall_mask_(y, x) = slit_open_[y] ? 1.0f : 0.0f;

    // Source position and phase
    const double tan_t = std::tan(math_stuff::deg2rad(p_.source_angle_deg));
    double yc_src = yc - (p_.x_wall - p_.x_src) * tan_t;
    yc_src = std::min(std::max(yc_src, static_cast<double>(p_.sponge)),
                      static_cast<double>(ny - p_.sponge - 1));
    src_y_ = static_cast<int>(std::nearbyint(yc_src));

    const double k = 2.0 * math_stuff::kPi / p_.wavelength;
    phase_offset_ = -k * (src_y_ - yc_src) * std::sin(math_stuff::deg2rad(p_.source_angle_deg));
    phase_noise_ = 0.0;
    omega_ = 2.0 * math_stuff::kPi / p_.wavelength;

    c_next_ = dampening_next_ * wall_mask_;
    c_prev_ = dampening_prev_ * dampening_next_ * wall_mask_;
    src_mask_ = wall_mask_(src_y_, p_.x_src);

    const double propagation_steps = (p_.x_det - p_.x_src) / static_cast<double>(p_.courant);
    settle_ = static_cast<int>(propagation_steps + 2 * AVG_STEPS);
    if (fresh)
        settle_ += static_cast<int>((p_.x_wall - p_.x_src) / static_cast<double>(p_.courant));
}

void WaveSimulator::reset()
{
    validateParams(p_);
    u_.fill(0.0f);
    u_prev_.fill(0.0f);
    u_next_.fill(0.0f);
    std::fill(I_det_.begin(), I_det_.end(), 0.0f);
    t_step_ = 0;
    rebuild(true);
}

float WaveSimulator::sourceValue(long long t)
{
    double phase = omega_ * p_.courant * static_cast<double>(t) + phase_offset_;
    if (!p_.coherent)
    {
        phase_noise_ += 0.1 * normal_(rng_);
        phase += phase_noise_;
    }
    const double amp = std::min(1.0, static_cast<double>(t) / 100.0);
    return static_cast<float>(amp * std::sin(phase) * src_mask_);
}

void WaveSimulator::step_n(int n)
{
    if (n <= 0)
        return;

    const float C2 = p_.courant * p_.courant;
    const int x0 = std::max(0, p_.x_det - p_.detector_width + 1);
    const int x1 = p_.x_det;
    const float inv_avg = 1.0f / static_cast<float>(AVG_STEPS);

    for (int k = 0; k < n; ++k)
    {
        // un = cn * (2u + C2 * lap(u)) - cp * up      (lap = 0 on the border)
        u_next_.assign(u_);
        u_next_ *= 2.0f;
        u_.laplacianInto(lap_);
        lap_ *= C2;
        u_next_ += lap_;
        u_next_ *= c_next_;
        u_next_.addcmul(c_prev_, u_prev_, -1.0f);

        u_next_(src_y_, p_.x_src) += sourceValue(t_step_ + k);

        u_next_.rowMeanSq(x0, x1, inst_);
        for (int y = 0; y < p_.ny; ++y)
            I_det_[y] += (inst_[y] - I_det_[y]) * inv_avg;

        // rotate: u -> u_next -> u_prev -> u   (O(1))
        u_prev_.swap(u_);
        u_.swap(u_next_);
    }

    t_step_ += n;
    settle_ = std::max(0, settle_ - n);
}