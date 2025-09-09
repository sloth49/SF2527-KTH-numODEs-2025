# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 1, Part 3-a
#
# Author: Alessio / Tim
# Date: 26 August 2025
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
from time import time


def forcing_func(z: np.ndarray, Q_0: float, a: float, b: float) -> np.ndarray:
    result = np.zeros_like(z)
    forced = np.where((z >= a) & (z <= b))
    result[forced] = Q_0 * np.sin((z[forced] - a) * np.pi / (b - a))
    return result


def alpha(v: float, alpha_0: float) -> float:
    return np.sqrt(0.25 * v**2 + alpha_0**2) - 0.5 * v


## PSEUDO CODE
# 1. Define the problem parameters
A = 0.1
B = 0.4
Q_0 = 7.0e3
ALPHA_0 = 50.0
T_OUT = 25.0
T_IN = 100.0
V = 1.0
# N_vals = [10, 20, 40, 80]
N_vals = [40]
L = 1.0

for N in N_vals:
    # define the grid
    z = np.linspace(start=0, stop=L, num=N+1)
    h = z[1] - z[0]
    # print(z); print(len(z)); print('h =', h)

# 2. Assemble the system matrix A and the right-hand side vector b

    q_of_z = forcing_func(z, Q_0, A, B)
    # plt.plot(z, q_of_z); plt.show()

    d_lo = -2 - V * h
    d_mid = 4
    d_up = -2 + V * h

    # internal points (j=2,...,N-1)
    diag_lo = np.full(N-3, d_lo)
    diag_mid = np.full(N-2, d_mid)
    diag_up = np.full(N-3, d_up)

    # inlet boundary (j=1)
    diag_mid = np.insert(diag_mid, 0, d_mid)
    diag_up = np.insert(diag_up, 0, d_up)

    # outlet boundary (j=N)
    diag_lo = np.append(diag_lo, -1, 1)
    diag_mid = np.append(diag_mid, 1, 1)
    
    # assemble matrix
    diagonals = [diag_lo, diag_mid, diag_up]
    A_matrix = diags(diagonals, offsets=[-1, 0, 1]).tocsc()


# 3. Solve the linear system
# 4. Plot the solution
# 5. Report the temperature values at centre for different grid sizes