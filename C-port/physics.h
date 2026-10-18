#pragma once
#include <utility>
#include <cstdint>
#include <stdexcept>
#include <random>
#include <string>
#include <vector>

class Tensor2D
{
public:
    Tensor2D() = default;
    Tensor2D(int rows, int cols, float value = 0.0f)
        : rows_(rows), cols_(cols)
    {
        if (rows < 0 || cols < 0)
            throw std::invalid_argument("tensor dimensions must be non-negative");
        d_.assign(static_cast<size_t>(rows),
                  std::vector<float>(static_cast<size_t>(cols), value));
    }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    std::pair<int, int> shape() const { return {rows_, cols_}; }

    float &operator()(int y, int x) { return d_[y][x]; }
    float operator()(int y, int x) const { return d_[y][x]; }
    std::vector<float> &operator[](int y) { return d_[y]; }
    const std::vector<float> &operator[](int y) const { return d_[y]; }
    void fill(float y);
    void swap(Tensor2D &other) noexcept
    {
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        d_.swap(other.d_);
    }

    void assign(const Tensor2D &other);

    Tensor2D &operator+=(const Tensor2D &other);
    Tensor2D &operator-=(const Tensor2D &other);
    Tensor2D &operator*=(const Tensor2D &other);
    Tensor2D &operator*=(float s);
    Tensor2D &operator+=(float s);
    /// this += alpha * a * b
    Tensor2D &addcmul(const Tensor2D &a, const Tensor2D &b, float alpha = 1.0f);
    Tensor2D matmul(const Tensor2D &other) const; // O(n^3)

    void laplacianInto(Tensor2D &out) const;
    Tensor2D laplacian() const
    {
        Tensor2D out(rows_, cols_);
        laplacianInto(out);
        return out;
    }

    void rowMeanSq(int x0, int x1, std::vector<float> &out) const;
    std::vector<float> flatten() const;

private:
    void requireSameShape(const Tensor2D &o, const char *what) const
    {
        if (rows_ != o.rows_ || cols_ != o.cols_)
            throw std::invalid_argument(std::string(what) + ": tensor shape mismatch");
    }
    int rows_ = 0,
        cols_ = 0;
    std::vector<std::vector<float>> d_;
};

inline Tensor2D operator+(Tensor2D a, const Tensor2D &b)
{
    a += b;
    return a;
}
inline Tensor2D operator-(Tensor2D a, const Tensor2D &b)
{
    a -= b;
    return a;
}
inline Tensor2D operator*(Tensor2D a, const Tensor2D &b)
{
    a *= b;
    return a;
}
inline Tensor2D operator*(Tensor2D a, float s)
{
    a *= s;
    return a;
}
inline Tensor2D operator*(float s, Tensor2D a)
{
    a *= s;
    return a;
}

struct Params
{
    int x_src = 60;
    int x_wall = 150;
    int x_det = 480;

    int nx = 1000;
    int ny = 400;

    int wall_thickness = 4;
    int sponge = 40;

    float courant = 0.4f;

    float wavelength = 12.0f;
    int n_slits = 2;
    float slit_width = 10.0f;
    float slit_spacing = 40.0f;
    std::string slit_shape = "rectangular";

    float source_angle_deg = 0.0f;
    float source_phase = 0.0f;
    float source_y_offset = 0.0f;
    bool coherent = true;
    bool particle_mode = false;

    int detector_oversampling = 4;
    int detector_width = 4;

    float wavelength_nm = 632.8f;
    float slit_width_um = 50.0f;
    float slit_spacing_um = 250.0f;
    float screen_distance_m = 1.5f;
    float screen_half_height_mm = 15.0f;

    float um_to_cells = 0.18f;
    float cells_per_m = 220.0f;
    float lam_cells_min = 8.0f;
    float lam_cells_max = 24.0f;

    float slit_height_um = 400.0f;
    float slit_height = 0.0f;
};

class WaveSimulator
{
public:
    static constexpr int AVG_STEPS = 480;
    explicit WaveSimulator(const Params &p, std::uint64_t seed = 0);
    void rebuild(bool fresh = false);
    void reset();
    void step_n(int n);
    void step()
    {
        step_n(1);
    }

    Params &params() { return p_; }
    const Params &params() const { return p_; }
    const Tensor2D &fieldSlice() const { return u_; }

    const std::vector<float> &detectorProfile() const { return I_det_; }
    const std::vector<float> &detectorImage() const { return I_det_; }
    const Tensor2D &wallMask() const { return wall_mask_; }
    const std::vector<unsigned char> &slitOpen() const { return slit_open_; }
    int srcY() const { return src_y_; }
    int nx() const { return p_.nx; }
    int ny() const { return p_.ny; }

    long tStep() const { return t_step_; }
    int settleRemaining() const { return settle_; }
    bool settled() const { return settle_ <= 0; }

private:
    void buildAbsorber();
    float sourceValue(long long t);
    Params p_;
    std::mt19937_64 rng_;
    std::normal_distribution<float> normal_{0.0f, 1.0f};
    long long t_step_ = 0;
    int settle_ = 0;
    Tensor2D u_, u_prev_, u_next_, lap_;
    std::vector<float> I_det_, inst_;

    Tensor2D dampening_prev_;
    Tensor2D dampening_next_;
    Tensor2D wall_mask_;
    Tensor2D c_next_, c_prev_;
    std::vector<unsigned char> slit_open_;
    std::vector<double> slit_centers_;

    int src_y_ = 0;
    double phase_offset_ = 0.0f;
    double phase_noise_ = 0.0f;
    double omega_ = 0.0;
    float src_mask_ = 1.0f;
};