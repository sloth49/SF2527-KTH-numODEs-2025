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
# from scipy.linalg import solve_banded
# from time import time


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
N_vals = [10, 20, 40, 80]
# N_vals = [40]
L = 1.0

# Prepare plotting area
fig, ax = plt.subplots(1, 1)

for N in N_vals:
    # define the grid
    z = np.linspace(start=0, stop=L, num=N+1)
    h = z[1] - z[0]
    # print(z); print(len(z)); print('h =', h)

# 2. Assemble the system matrix A and the right-hand side vector b

    q_of_z = forcing_func(z, Q_0, A, B)
    # plt.plot(z, q_of_z); plt.show()
    alpha_of_v = alpha(v=V, alpha_0=ALPHA_0)

    d_lo = -2 - V * h
    d_mid = 4
    d_up = -2 + V * h
    an1 = -2 -V * h -2 + V* h
    an = 4 - 2 * alpha_of_v * h * (-2 + V * h)
    b1 = 2 * h**2 * q_of_z[1] + (2 + V * h) * T_IN
    bn = -2 * alpha_of_v * h * (-2 + V * h)

    # Main matrix diagonals
    diag_lo = np.full(shape=N-1, fill_value=d_lo)
    diag_lo[-1] = an1
    diag_mid = np.full(shape=N, fill_value=d_mid)
    diag_mid[-1] = an
    diag_up = np.full(shape=N-1, fill_value=d_up)

    # RHS of linear system
    b_vector = 2 * h**2 * q_of_z[1:]
    b_vector[-1] = bn
    
    # assemble matrix
    diagonals = [diag_lo, diag_mid, diag_up]
    A_matrix = diags(diagonals, offsets=[-1, 0, 1]).tocsc()

# 3. Solve the linear system
    T = spsolve(A=A_matrix, b=b_vector)

# 4. Plot the solution
    ax.plot(z, np.concatenate(([T_IN], T)), label=f'N={N}')

# 5. Report the temperature values at centre for different grid sizes
    # T_mid = 
    print()

# Finalise the plot
plt.legend()
plt.title('$T(z)$')
plt.grid()
plt.show()