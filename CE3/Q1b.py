# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 3, Part 1-b
#
# Author: Alessio / Tim
# Date: 14 October 2025
# -----------------------------------------------------------------------------

# Plot the PDE amplification factor with the advection speed a as
# a parameter, to represent the stability condition for Lax-Friedrichs' method

import numpy as np
import matplotlib.pyplot as plt

def g(z, a):
    gx = np.cos(z)
    gy = a * np.sin(z)
    return gx, gy

z = np.linspace(0, 2*np.pi)
Co_vals = [0.5, 0.75, 1.0, 1.25, 1.5]

fig = plt.figure()
for Co in Co_vals:
    gx, gy = g(z, Co)
    if Co ==1.0:
        plt.plot(gx, gy, 'k--', label=f'Co = {Co:.1f}')
    else:
        plt.plot(gx, gy, label=f'Co = {Co:.1f}')
plt.grid()
plt.axis('equal')
plt.legend()
plt.show()