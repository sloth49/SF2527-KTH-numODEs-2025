# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 1, Part 2-b
#
# Author: Alessio / Tim
# Date: 26 August 2025
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Rate constants
r1 = 5.0e-2
r2 = 1.2e4
r3 = 4.0e7

# Define the ODE system for Robertson's problem
def robertson_rhs(t,u):
    xA, xB, xC = u
    dxA_dt = -r1 * xA + r2 * xB * xC
    dxB_dt = r1 * xA - r2 * xB * xC - r3 * xB ** 2
    dxC_dt = r3 * xB ** 2
    return np.array([dxA_dt, dxB_dt, dxC_dt])

# Jacobian matrix for Robertson's problem
def robertson_jacobian(t, u):
    xA, xB, xC = u
    J = np.array([
        [-r1, r2 * xC, r2 * xB],
        [r1, -r2 * xC - 2 * r3 * xB, -r2 * xB],
        [0, 2 * r3 * xB, 0]
    ])
    return J

# Runge-Kutta method from Part 1
def RK_step(f, u, t, h):
    k1 = f(t, u)
    k2 = f(t + h, u + h * k1)
    k3 = f(t + h/2, u + h/4 * (k1 + k2))
    return u + (h / 6) * (k1 + k2 + 4 * k3)

def integrate(f, u0, t_span, h):
    t = 0
    u = u0.copy()
    n_steps = int(T / h)
    
    start_time = time.time()
    
    for i in range(n_steps):
        u = RK_step(f, u, t, h)
        t += h
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    return u, computation_time

# Initial conditions
u0 = np.array([1.0, 0.0, 0.0])
T = 50

stable_h = 5e-7

print(f"Solving Robertson's problem for t ∈ [0, {T}] with h = {stable_h}")

final_u, comp_time = integrate(robertson_rhs, u0, 1000, stable_h)

xA_final = final_u[0]
xB_final = final_u[1]
xC_final = final_u[2]

print("\nResults:")
print(f"Stepsize used: h = {stable_h:.2e}")
print(f"Number of steps: {int(1000/stable_h):,}")
print(f"Computation time: {comp_time:.4f} seconds")
print(f"Final values at t = {T}:")
print(f"  xA(1000) = {xA_final:.12f}")
print(f"  xB(1000) = {xB_final:.12f}")
print(f"  xC(1000) = {xC_final:.12f}")




