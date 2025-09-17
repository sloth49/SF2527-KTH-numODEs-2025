# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 1, Part 1-a
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
    return u + (h / 6.0) * (k1 + k2 + 4.0 * k3)


def integrate(u0: np.ndarray, a: np.ndarray, h: float, Tsteps: int) -> np.ndarray:
    u = np.zeros((u0.shape[0], Tsteps+1))
    for k in range(Tsteps+1):
        if k == 0:
            u[:, k] = u0
        else:
            u[:, k] = RK_step(u[:, k-1], a, h)
    return u


def plot_components(t: np.ndarray, m: np.ndarray) -> None:
    plt.plot(t, m[0, :], label='$m_1$')
    plt.plot(t, m[1, :], label='$m_2$')
    plt.plot(t, m[2, :], label='$m_3$')
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
    ax.quiver(0, 0, 0, a[0], a[1], a[2], length=1.0, normalize=True,
              color='r', linewidth=2, label='$a$ vector')    
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('$m_1$')
    ax.set_ylabel('$m_2$')
    ax.set_zlabel('$m_3$')
    ax.legend()
    plt.show()


# -------------------------------- Main --------------------------------------

# Initial data / setup
a = 0.25 * np.array([1, np.sqrt(11), 2])
m0 = np.array([0.0, 0.0, 1.0])

# Time integration
Tf = 50.0
Tsteps = 200
t = np.linspace(start=0, stop=Tf, num=Tsteps+1)
h = t[1] - t[0]
print(f'Time step h = {h:.4f}')
m = integrate(m0, a, h, Tsteps)

# Plot components vs time
plot_components(t, m)

# Plot trajectory in 3D
plot_trajectory(m, a)