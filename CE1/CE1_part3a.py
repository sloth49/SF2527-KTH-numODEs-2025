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
import math


def forcing_func(z: np.ndarray, Q_0: float, a: float, b: float) -> np.ndarray:
    result = np.zeros_like(z)
    forced = np.where((z >= a) & (z <= b))
    result[forced] = Q_0 * np.sin((z[forced] - a) * np.pi / (b - a))
    return result


def alpha(v: float, alpha_0: float) -> float:
    return math.sqrt(0.25 * v**2 + alpha_0**2) - 0.5 * v


# Define the problem parameters
A = 0.1
B = 0.4
Q_0 = 7.0e3
ALPHA_0 = 50.0
T_IN = 100.0
T_OUT = 25.0
V = 1.0
N_VALS = [10, 20, 40, 80, 160, 320]
L = 1.0
Z_PROBE = 0.5

# Prepare plotting area
fig, ax = plt.subplots(1, 1)

for N in N_VALS:
    # define the grid
    z = np.linspace(start=0, stop=L, num=N+1)
    h = z[1] - z[0]

    # Assemble the system matrix A and the right-hand side vector b
    q_of_z = forcing_func(z, Q_0, A, B)
    alpha_of_v = alpha(v=V, alpha_0=ALPHA_0)

    d_lo = -(V/(2*h) + 1/h**2)
    d_mid = 2/h**2
    d_up = V/(2*h) - 1/h**2
    diag_lo = np.full(shape=N-1, fill_value=d_lo)
    diag_mid = np.full(shape=N, fill_value=d_mid)
    diag_up = np.full(shape=N-1, fill_value=d_up)

    b_vector = q_of_z[1:]

    # Impose BCs
    an1 = -2/h**2
    an = 2/h**2 - alpha_of_v * (V - 2/h)
    diag_lo[-1] = an1
    diag_mid[-1] = an
    b1 = q_of_z[1] + (1/h**2 + V/(2*h)) * T_IN
    bn = q_of_z[-1] - alpha_of_v * (V - 2/h) * T_OUT
    b_vector[0] = b1
    b_vector[-1] = bn
    
    # assemble matrix
    diagonals = [diag_lo, diag_mid, diag_up]
    A_matrix = diags(diagonals, offsets=[-1, 0, 1]).tocsc()

    # Solve the linear system
    T = spsolve(A=A_matrix, b=b_vector)

    # Plot the solution
    T = np.concatenate(([T_IN], T))
    ax.plot(z, T, label=f'N={N}')

    # Report the temperature values at centre
    if N in [80, 160, 320]:
        T_mid = np.interp(x=Z_PROBE, xp=z, fp=T)
        print(f'N={N}, T({Z_PROBE}) = {T_mid:.6f}')

# Finalise the plot
plt.legend()
plt.title('$T(z)$')
plt.xlabel('$z$')
plt.grid()
plt.show()