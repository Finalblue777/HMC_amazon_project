import re
import multiprocess
import numpy as np

def func(x):
    return np.random.rand()


class fd_grad(object):
    def __init__(self, func, x, fd_eps=1e-5):
        """Create a gradient object"""
        self.x = np.asarray(x).flatten()
        self.e = np.zeros_like(self.x)
        self.func = func
        self.fd_eps = fd_eps

    def __call__(self, i):
        """Evaluate and return ith entry of the gradient"""
        self.e[...] = 0
        self.e[i] = self.fd_eps
        fd = (self.func(self.x+self.e) - self.func(self.x)) / self.fd_eps
        print(">>>i, fd>>>", i, fd)
        return fd


if __name__ == "__main__":
    with multiprocess.Pool(3) as pool:
        print(pool.map(fd_grad(func=func,
                               x=[10, 3, 12, 44], ), range(4) ))
