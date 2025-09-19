# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 1, Part 2-e
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
def robertson_rhs(t, u):
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

def RK_step(f, u, t, h):
    k1 = f(t, u)
    k2 = f(t + h, u + h * k1)
    k3 = f(t + h/2, u + h/4 * (k1 + k2))
    return u + (h / 6) * (k1 + k2 + 4 * k3)

def integrate_RK(f, u0, T, h):
    t = 0
    u = u0.copy()
    n_steps = int(T / h)

    start_time = time.time()
    for i in range(n_steps):
        u = RK_step(f, u, t, h)
        t += h
    computation_time = time.time() - start_time
    return u, computation_time

# Implicit Euler step using Newton's method
def IE_step(f, jac, u, t, h, tol=1e-8, max_iter=100):
    # Solve: u_{n+1} = u_n + h * f(t_{n+1}, u_{n+1})
    # Let F(u) = u - u_n - h * f(t+h, u) = 0
    u_next = u.copy()  # initial guess
    for i in range(max_iter):
        F = u_next - u - h * f(t+h, u_next)
        if np.linalg.norm(F) < tol:
            break
        J = np.eye(3) - h * jac(t+h, u_next)
        delta = np.linalg.solve(J, -F)
        u_next += delta
    return u_next

def integrate_IE(f, jac, u0, t_span, h):
    t0, tf = t_span
    n_steps = int((tf - t0) / h)
    u = u0.copy()
    t = t0
    for i in range(n_steps):
        u = IE_step(f, jac, u, t, h)
        t += h
    return u

# Initial conditions
u0 = np.array([1.0, 0.0, 0.0])
t_span = (0, 1000)

# Accurate values
accurate_values = np.array([0.293414227164, 0.000001716342048, 0.706584056494])

# RK with h=1e-4
h_RK = 1e-4
start_time = time.time()
u_final_RK, time_RK = integrate_RK(robertson_rhs, u0, 1000, h_RK)
error_RK = np.linalg.norm(u_final_RK - accurate_values)

# IE with different step sizes
h_IE_list = [0.1, 0.05, 0.01, 0.005]
results_IE = []
for h_IE in h_IE_list:
    start_time = time.time()
    u_final_IE = integrate_IE(robertson_rhs, robertson_jacobian, u0, (0,1000), h_IE)
    time_IE = time.time() - start_time
    error_IE = np.linalg.norm(u_final_IE - accurate_values)
    results_IE.append((h_IE, error_IE, time_IE))

# Print table
print("Efficiency Comparison:")
print("Method\t\th\t\tError\t\tTime (s)")
print(f"RK\t\t{h_RK}\t{error_RK}\t{time_RK}")
for h_IE, error_IE, time_IE in results_IE:
    print(f"IE\t\t{h_IE}\t{error_IE}\t{time_IE}")