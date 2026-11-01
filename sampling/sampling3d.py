import numpy as np


class ParticleSampler3D:
    """
    Monte Carlo sampler for particle detection events in a 3D simulation based on a given intensity distribution.
    """

    def __init__(self, ny, nz, x_detector, max_recent=5000):
        self.ny, self.nz = ny, nz
        self.x_detector = x_detector
        self.max_recent = max_recent
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self):
        """
        Resets the sampler's internal state, clearing the histogram and recent samples."""
        self.hist = np.zeros((self.ny, self.nz))
        self.recent = np.empty((0, 2))

    def sample(self, intensity, k):
        """
        Samples particle detection events based on the given intensity distribution."""
        k = self.rng.poisson(k)
        if k == 0:
            return
        flat = np.asarray(intensity, dtype=np.float64).ravel()
        cdf = np.cumsum(flat)
        if cdf[-1] <= 1e-12:
            return
        cdf /= cdf[-1]
        idx = np.minimum(np.searchsorted(cdf, self.rng.random(k)), flat.size - 1)
        iy, iz = np.divmod(idx, self.nz)
        self.hist += np.bincount(idx, minlength=flat.size).reshape(self.ny, self.nz)
        pts = np.column_stack(
            [iy + self.rng.random(k) - 0.5, iz + self.rng.random(k) - 0.5]
        )
        self.recent = np.vstack([self.recent, pts])[-self.max_recent:]