import numpy as np
from numpy import random as rnd
import copy


class OUNoise():
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = rnd.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * rnd.randn(len(x))
        self.state = x + dx
        return self.state
