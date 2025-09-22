# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 1, Part 2-d
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

# Implicit Euler step using Newton's method
def IE_step(f, jac, u, t, h, tol=1e-10, max_iter=100):
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

# Test with h=0.1 for stability (for t in [0,10])
h_IE = 0.1
print("Testing IE for t in [0,10] with h=0.1")
u_final_IE_test = integrate_IE(robertson_rhs, robertson_jacobian, u0, (0,10), h_IE)
print(f"Solution at t=10: {u_final_IE_test}")

# Now run for t in [0,1000] with h=0.1
start_time = time.time()
u_final_IE = integrate_IE(robertson_rhs, robertson_jacobian, u0, t_span, h_IE)
end_time = time.time()
computational_time_IE = end_time - start_time

# Accurate values at t=1000
accurate_values = np.array([0.293414227164, 0.000001716342048, 0.706584056494])
error_IE = np.linalg.norm(u_final_IE - accurate_values)

print(f"Part 2(d): IE method for t in [0,1000]")
print(f"Step size h = {h_IE}")
print(f"Computed solution at t=1000: xA = {u_final_IE[0]}, xB = {u_final_IE[1]}, xC = {u_final_IE[2]}")
print(f"Error (Euclidean norm) = {error_IE}")
print(f"Computational time = {computational_time_IE} seconds")