import numpy as np

class ParticleSampler:
    def __init__(self,ny,x_detector,max_recent=5000):
        self.ny,self.x_detector,self.max_recent = ny,x_detector,max_recent
        self.rng = np.random.default_rng()
        self.reset()
    def reset(self):
        self.hist = np.zeros(self.ny)
        self.recent = np.empty((0,2))
    def sample(self,intensity,k):
        k = self.rng.poisson(k)
        if k == 0:
            return np.empty(0)
        cdf = np.cumsum(intensity,dtype=np.float32)
        if(cdf[-1] <= 1e-12):
            return np.empty(0)
        cdf /= cdf[-1]
        idx = np.minimum(np.searchsorted(cdf,self.rng.random(k)),self.ny-1)
        y = idx + self.rng.random(k) - 0.5
        self.hist += np.bincount(idx,minlength=self.ny)
        pts = np.column_stack([self.x_detector + self.rng.uniform(0,8,k),y])
        self.recent = np.vstack([self.recent,pts])[-self.max_recent:]
        return y