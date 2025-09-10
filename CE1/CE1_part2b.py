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

t, u, jac_eigvals = integrate(robertson_rhs, u0, t_span, stable_h)

eigvals_real = jac_eigvals.real.flatten()
eigvals_imag = jac_eigvals.imag.flatten()

plt.figure(figsize=(10, 6))
plt.scatter(eigvals_real, eigvals_imag, s=4)
plt.xlabel('Re(λ)')
plt.ylabel('Im(λ)')
plt.title('Jacobian Eigenvalues in the Complex Plane')
plt.grid()
plt.show()

non_zero_eigvals = np.zeros((2, len(t)), dtype=complex)
for i in range(len(t)):
    eigvals = jac_eigvals[:, i]
    # Sort by absolute value and take the two largest
    sorted_indices = np.argsort(np.abs(eigvals))[::-1]
    non_zero_eigvals[:, i] = eigvals[sorted_indices[:2]]

#Test for some indices
middle_indices = [len(t)//4, len(t)//2, 3*len(t)//4]
for i in middle_indices:
    eig1, eig2, eig3 = jac_eigvals[:, i]
    print(f"{t[i]:.6f}\t{eig1:.6e}\t{eig2:.6e}\t{eig3:.6e}")

# Plot the real and imaginary parts of the non-zero eigenvalues over time
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, non_zero_eigvals[0, :].real, label='Largest eigenvalue (real)')
plt.xlabel('Time t')
plt.ylabel('Re(λ)')
plt.legend()
plt.grid()
plt.title('Real Parts of the largest eigenvalue vs Time')

plt.subplot(2, 1, 2)
plt.plot(t, non_zero_eigvals[1, :].real, label='Second largest eigenvalue (real)')
plt.xlabel('Time t')
plt.ylabel('Re(λ)')
plt.legend()
plt.grid()
plt.title('Real Parts of second largest eigenvalue vs Time')

#Plot of the imaginary part
#plt.subplot(3, 1, 3)
#plt.plot(t, non_zero_eigvals[0, :].imag, label='Largest eigenvalue (imag)')
#plt.plot(t, non_zero_eigvals[1, :].imag, label='Second largest eigenvalue (imag)')
#plt.ylabel('Im(λ)')
#plt.xlabel('Time t')
#plt.legend()
#plt.title('Imaginary Parts of Non-Zero Jacobian Eigenvalues vs Time')
#plt.grid()

plt.tight_layout()
plt.show()

# Also, plot the magnitude of the largest eigenvalue over time
plt.figure(figsize=(10, 6))
largest_magnitude = np.abs(non_zero_eigvals[0, :])
plt.plot(t, largest_magnitude, label='Magnitude of largest eigenvalue')
plt.xlabel('Time t')
plt.ylabel('|λ|')
plt.yscale('log')  # Use log scale due to large range
plt.grid()
plt.title('Magnitude of Largest Eigenvalue vs Time')
plt.legend()
plt.show()

