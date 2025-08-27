# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 1, Part 1-b
#
# Author: Alessio / Tim
# Date: 26 August 2025
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# RHS of the ODE
def f_rhs(u: np.ndarray, a: np.ndarray) -> np.ndarray:
    ALPHA = 0.07
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


def plot_components(t: np.ndarray, m: np.ndarray) -> None:
    plt.plot(t, m[0, :], label='m1')
    plt.plot(t, m[1, :], label='m2')
    plt.plot(t, m[2, :], label='m3')
    plt.xlabel('t') 
    plt.ylabel('m')
    plt.legend()   
    plt.grid() 
    plt.title('Magnetization components over time')
    plt.show()


def plot_trajectory(m: np.ndarray, a: np.ndarray) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(m[0, :], m[1, :], m[2, :], label='Trajectory')
    ax.quiver(0, 0, 0, a[0], a[1], a[2], 
              color='r', linewidth=2, label='$a$ vector')    
    ax.set_box_aspect([1,1,1])
    ax.legend()
    plt.show()


# -------------------------------- Main --------------------------------------

# Initial data / setup
a = 0.25 * np.array([1, np.sqrt(11), 2])
m0 = np.array([0, 0, 1])

Tf = 5
Tsteps = [50, 100, 200, 400, 800, 1600]

# Determine order of accuracy - See 'Notes on Order of Accuracy' section 3
mn_vals = []
h_vals = []
deltas = []
 
for Tstep in Tsteps:
    h = Tf / Tstep
    h_vals.append(h)
    m = integrate(m0, a, h, Tstep)
    mn_vals.append(m[:, -1])
    if Tstep != Tsteps[0]:
        delta = np.linalg.norm(mn_vals[-1] - mn_vals[-2])
        deltas.append(delta)

plt.loglog(h_vals[:-1], deltas, marker='o', label='Deltas')
plt.loglog(h_vals, [h_val**3 for h_val in h_vals], linestyle='--', label='$O(h)$')
plt.xlabel('$h$')
plt.ylabel('$|m_N(T) - m_{2N}(T)|$')
plt.title('Order of Accuracy')
plt.legend()
plt.grid()
plt.show()