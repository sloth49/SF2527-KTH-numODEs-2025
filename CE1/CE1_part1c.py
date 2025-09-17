# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 1, Part 1-c
#
# Author: Alessio / Tim
# Date: 26 August 2025
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ALPHA = 0.07

# RHS of the ODE
def f_rhs(u: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.cross(a, u) + ALPHA * np.cross(a, np.cross(a, u))


# Runge-Kutta step
def RK_step(u: np.ndarray, a: np.ndarray, h: float) -> np.ndarray:
    k1 = f_rhs(u, a)
    k2 = f_rhs(u + h * k1, a)
    k3 = f_rhs(u + 0.25 * h * (k1 + k2), a)
    return u + (h / 6) * (k1 + k2 + 4 * k3)


def integrate(u0: np.ndarray, a: np.ndarray, h: float, Tsteps: int) -> np.ndarray:
    u = np.zeros((u0.shape[0], Tsteps + 1))
    for k in range(Tsteps + 1):
        if k == 0:
            u[:, k] = u0
        else:
            u[:, k] = RK_step(u[:, k-1], a, h)
    return u


def assembleA(a: np.ndarray) -> np.ndarray:
    a1, a2, a3 = a
    A = np.array([
        [
            -ALPHA * (a2**2 + a3**2),
            ALPHA * a1 * a2 - a3,
            ALPHA* a1 * a3 + a2
        ],
        [
            ALPHA * a1 * a2 + a3,
            -ALPHA * (a1**2 + a3**2),
            ALPHA * a2 * a3 - a1
        ],
        [
            ALPHA * a1 * a3 - a2,
            ALPHA * a2 * a3 + a1,
            -ALPHA * (a1**2 + a2**2)
        ]
    ])
    return A


# Compute eigenvalues of A
def eigvalsA(a: np.ndarray) -> np.ndarray:
    A = assembleA(a)
    return np.linalg.eigvals(A)


# Define the function |R(z=x+iy)| - 1
def stability_boundary(x, y):
    z = x + 1j * y  # Convert to complex number
    return np.abs(1 + z + (1/2) * z**2 + (1/6) * z**3) - 1


def plot_stability_region(eigvals: np.ndarray) -> None:
    x = np.linspace(-4, 2, 400)
    y = np.linspace(-4, 4, 400)
    X, Y = np.meshgrid(x, y)
    Z = stability_boundary(X, Y)

    plt.figure(figsize=(6, 6))
    plt.contourf(X, Y, Z, levels=[-1, 0], colors=['lightblue'])  # Fill interior region
    plt.contour(X, Y, Z, levels=[0], colors='b')  # Boundary contour
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title("Stability Region of the RK Method")
    plt.axis('equal')

    # Add markers for the two complex eigenvalues of A
    for eigval in eigvals:
        if np.imag(eigval) != 0:
            plt.plot(np.real(eigval), np.imag(eigval), 'ro', markersize=8, label='Eigenvalue')
            # Scale the eigenvalue by a large factor to draw the line
            multiplier = 4
            plt.plot(
                [0, np.real(eigval) * multiplier],
                [0, np.imag(eigval) * multiplier],
                'r--', linewidth=1)
    plt.legend()
    plt.show()


def h_absolute_stability(eigvals: np.ndarray) -> float:
    h_values = []
    for eigval in eigvals:
        x, y = np.real(eigval), np.imag(eigval)
        if y != 0:
            # Solve for h such that (h*x, h*y) is on the stability boundary
            def objective(h):
                return stability_boundary(h * x, h * y)
            from scipy.optimize import fsolve
            try:
                h_root = fsolve(func=objective, x0=3.0)[0]
                if h_root > 0:  # avoid trivial root at h=0
                    h_values.append(h_root)
            except ValueError:
                pass  # handle case where no root is found
    return min(h_values)


# -------------------------------- Main --------------------------------------

# Initial data / setup
a = 0.25 * np.array([1, np.sqrt(11), 2])
m0 = np.array([0, 0, 1])

eigvals = eigvalsA(a)
print("Eigenvalues of A:\n", eigvals)

plot_stability_region(eigvals)

h0 = h_absolute_stability(eigvals)
print('h0 =', h0)