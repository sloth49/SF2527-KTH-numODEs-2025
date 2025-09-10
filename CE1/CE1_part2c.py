# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 1, Part 2-b
#
# Author: Alessio / Tim
# Date: 26 August 2025
# -----------------------------------------------------------------------------

import numpy as np
import time

# Rate constants
r1 = 5.0e-2
r2 = 1.2e4
r3 = 4.0e7

# Define the ODE system for Robertson's problem
def robertson_rhs(u):
    xA, xB, xC = u
    dxA_dt = -r1 * xA + r2 * xB * xC
    dxB_dt = r1 * xA - r2 * xB * xC - r3 * xB ** 2
    dxC_dt = r3 * xB ** 2
    return np.array([dxA_dt, dxB_dt, dxC_dt])


# Runge-Kutta method from Part 1
def RK_step(f, u, h):
    k1 = f(u)
    k2 = f(u + h * k1)
    k3 = f(u + h/4 * (k1 + k2))
    return u + (h / 6) * (k1 + k2 + 4 * k3)


def integrate(f, u0, t_span, h):
    t0, tf = t_span
    n_steps = int((tf - t0) / h)
    u = u0.copy()
    
    start_time = time.time()
    for n in range(n_steps):
        u += RK_step(f, u, h)
    end_time = time.time()
    computation_time = end_time - start_time
    
    return u, computation_time


# Initial conditions
u0 = np.array([1.0, 0.0, 0.0])
T0 = 0.0
TF = 1000.0
h = 1e-5

print(f"Solving Robertson's problem for t ∈ [0, {TF:.0f}] with h = {h}")

final_u, comp_time = integrate(
    f=robertson_rhs, u0=u0, t_span=(T0, TF), h=h)

xA_final, xB_final, xC_final = final_u

print("\nResults:")
print(f"Stepsize used: h = {h:.2e}")
print(f"Number of steps: {int(1000/h):,}")
print(f"Computation time: {comp_time:.4f} seconds")
print(f"Final values at t = {TF}:")
print(f"  xA(1000) = {xA_final:.12f}")
print(f"  xB(1000) = {xB_final:.12f}")
print(f"  xC(1000) = {xC_final:.12f}")