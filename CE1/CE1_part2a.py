# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 1, Part 2-a
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
    
    for i in range(n_steps):
        u[:, i+1] = RK_step(f, u[:, i], t[i], h)
    
    return t, u

#Solve with RK method for t in [0, 10]
print("Part 2a: Solving with RK method for t in [0, 10]")

# Initial conditions
u0 = np.array([1.0, 0.0, 0.0])
t_span = (0, 10)

# Find stable step size empirically
h_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
stable_h = None

for h in h_values:
    try:
        t, u = integrate(robertson_rhs, u0, t_span, h)
        # Check if solution is stable (no NaN or extreme values)
        if not np.any(np.isnan(u)) and np.all(np.abs(u) < 1e10): 
            stable_h = h
            print(f"Stable solution found with h = {h}")
            break
    except:
        continue

if stable_h is None:
    print("No stable solution found with tested step sizes")
    stable_h = 1e-5  # Use smallest tested value

# Solve with stable step size
t, u = integrate(robertson_rhs, u0, t_span, stable_h)
xA, xB, xC = u

# Plot solutions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, xA, label='$x_A$')
plt.plot(t, xB, label='$x_B$')
plt.plot(t, xC, label='$x_C$')
plt.xlabel('t')
plt.ylabel('Concentration')
plt.title('Robertson Problem - Linear Scale')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.semilogy(t, xA, label='$x_A$')
plt.semilogy(t, xB, label='$x_B$')
plt.semilogy(t, xC, label='$x_C$')
plt.xlabel('t')
plt.ylabel('Concentration (log scale)')
plt.title('Robertson Problem - Log Scale')
plt.legend()
plt.grid()

plt.show()

print(f"Empirically found stable step size: h = {stable_h}")