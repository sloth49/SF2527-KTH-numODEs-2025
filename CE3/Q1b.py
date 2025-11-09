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

def g(t, a):
    x = np.cos(t)
    y = a * np.sin(t)
    return x, y

t = np.linspace(0, 2*np.pi)
a_vals = [0.5, 0.75, 1.0, 1.25, 1.5]

fig = plt.figure()
for a in a_vals:
    x, y = g(t, a)
    if a ==1.0:
        plt.plot(x, y, 'k--', label=f'$a$ = {a:.1f}')
    else:
        plt.plot(x, y, label=f'$a$ = {a:.1f}')
plt.grid()
plt.axis('equal')
plt.legend()
plt.show()