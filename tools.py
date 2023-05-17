import numpy as np


def standardize(v):
    '''
    Method to standardize a rank-1 np array
    https://github.com/kengz/SLM-Lab/blob/master/slm_lab/lib/math_util.py#LL23C1-L27C17
    '''
    assert len(v) > 1, 'Cannot standardize vector of size 1'
    v_std = (v - v.mean()) / (v.std() + 1e-08)
    return v_std


class RewardScaler:
    def __init__(self, dim, gamma, max_iter=1e+20, eps=1e-8):
        '''PPO scaling optimization (https://arxiv.org/pdf/2005.12729.pdf)'''
        self.mean = np.zeros(dim, dtype=np.float32)
        self.M2 = np.zeros(dim, dtype=np.float32)
        self.R = np.zeros(dim, dtype=np.float32)
        self.n = 0
        self.gamma = gamma
        self.max_iter = max_iter
        self.eps = eps


    def step(self, r):
        self.R = self.R * self.gamma + r
        self.n += 1

        delta = self.R - self.mean
        self.mean += delta / self.n
        delta2 = self.R - self.mean
        self.M2 += delta * delta2

        var = self.M2 / self.n
        var = np.where(var <= 0, self.eps, var)
        std = np.sqrt(var)

        return np.clip(r / std, -2, 2)
