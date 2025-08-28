import numpy as np

import numpy as np
import math as mt
import time as tm
rng=np.random


def random_gauss_sqrmat(d):
    fA = rng.standard_normal((d, d))
    return fA

rnscl = .05
def val_sqrmat(x,A): return np.sum((x @ A) * x, axis=1)  # + rng.standard_normal((x.shape[0],))*rnscl
def grad_sqrmat(x,A): return (A+A.T) @ x


# --- Rosenbrock (hard version)

def rosenbrock(x: np.ndarray) -> np.ndarray: return np.sum(
    100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, axis=1)


rosen_grad = lambda x: np.pad(-400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1]), (0, 1)) + np.pad(
    200 * (x[1:] - x[:-1] ** 2), (1, 0))


