import numpy as np

import numpy as np
import math as mt
import time as tm
rng=np.random


# --- Random d-dimensional twice differential surface.
fA = None
gA = None


def random_gauss_surface(d):
    global fA, gA
    fA = rng.standard_normal((d, d), dtype=np.float64)
    gA = fA + fA.T



rnscl = .05


def mat_value(x): return np.sum((x @ fA) * x, axis=1)  # + rng.standard_normal((x.shape[0],))*rnscl


def mat_grad(x): return gA @ x


# --- Rosenbrock (hard version)

def rosenbrock(x: np.ndarray) -> np.ndarray: return np.sum(
    100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, axis=1)


rosen_grad = lambda x: np.pad(-400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1]), (0, 1)) + np.pad(
    200 * (x[1:] - x[:-1] ** 2), (1, 0))

# --- Assign value and gradient namespace
value = mat_value
grad = mat_grad
