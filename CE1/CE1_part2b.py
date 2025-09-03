# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 1, Part 2-a
#
# Author: Alessio / Tim
# Date: 26 August 2025
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

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
        [-r1, r2 * xC,                r2 * xB],
        [r1,  -r2 * xC - 2 * r3 * xB, -r2 * xB],
        [0,   2 * r3 * xB,            0]
    ])
    return J

# Runge-Kutta method from Part 1
def RK_step(f, u, t, h):
    k1 = f(t, u)
    k2 = f(t + h, u + h * k1)
    k3 = f(t + h/2, u + h/4 * (k1 + k2))
    return u + (h / 6) * (k1 + k2 + 4 * k3)

def integrate(f, u0, t_span, h):
    t0, tf = t_span
    n_steps = int((tf - t0) / h)
    t = np.linspace(t0, tf, n_steps + 1)
    u = np.zeros((len(u0), n_steps + 1))
    u[:, 0] = u0

    jac_eigvals = np.zeros((len(u0), n_steps + 1), dtype=complex)
    jac_eigvals[:, 0] = np.linalg.eigvals(robertson_jacobian(t0, u0))
    
    for i in range(n_steps):
        u[:, i+1] = RK_step(f, u[:, i], t[i], h)
        jac = robertson_jacobian(t[i], u[:, i+1])
        jac_eigvals[:, i+1] = np.linalg.eigvals(jac)
    
    return t, u, jac_eigvals

# Initial conditions
u0 = np.array([1.0, 0.0, 0.0])
t_span = (0, 10)
stable_h = 7.5e-4

# Solve with stable step size, return Jacobian eigenvalues for each time step
t, u, jac_eigvals = integrate(robertson_rhs, u0, t_span, stable_h)

# # Plot solutions
# plt.figure(figsize=(12, 5))
# xA, xB, xC = u

# plt.subplot(1, 2, 1)
# plt.plot(t, xA, label='$x_A$')
# plt.plot(t, xB, label='$xB$')
# plt.plot(t, xC, label='$xC$')
# plt.xlabel('t')
# plt.ylabel('Concentration')
# plt.title('Robertson Problem - Linear Scale')
# plt.legend()
# plt.grid()

# plt.subplot(1, 2, 2)
# plt.semilogy(t, xA, label='$xA$')
# plt.semilogy(t, xB, label='$xB$')
# plt.semilogy(t, xC, label='$xC$')
# plt.xlabel('t')
# plt.ylabel('Concentration (log scale)')
# plt.title('Robertson Problem - Log Scale')
# plt.legend()
# plt.grid()

# plt.show()

# Plot all eigenvalues at once for the first N time steps
# N = 2500
# eigvals_real = jac_eigvals[:, :].real.flatten()
# eigvals_imag = jac_eigvals[:, :].imag.flatten()
eigvals_real = jac_eigvals.real.flatten()
eigvals_imag = jac_eigvals.imag.flatten()

plt.figure(figsize=(10, 6))
plt.scatter(eigvals_real, eigvals_imag, s=4)
plt.xlabel('Re(λ)')
plt.ylabel('Im(λ)')
plt.title('Jacobian Eigenvalues in the Complex Plane')
plt.grid()
plt.show()

# TODO plot real part vs time
