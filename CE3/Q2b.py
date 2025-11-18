# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 3, Part 2-c
#
# Author: Alessio / Tim
# Date: 3 November 2025
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from domain import Domain
from plot_util import save2jpg
import os


def forcing_integral(xn, tn, char_speed, s, r, t0=0, N=400):
    """
    Integrates the function
          s(csi(tau)) * r(tau) * dtau
    between tau = [t0, tn], using Simpson's rule
    """
    if tn == 0:
        return 0.0
    tau = np.linspace(start=t0, stop=tn, num=N)
    csi = xn - char_speed * (tn - tau)
    integrand = s(csi) * r(tau)
    return simpson(y=integrand, x=tau)


def main(savefig = False):
     """
     Computes the analytical solution and saves results to file.
     The file is a dictionary-type archive, with 'arr_0', 'arr_1', ... keys
     to retrieve the solution components. Uses Simpson's rule on discretised
     values to calculate the integral on the RHS of the equation.
     """
     # Problem parameters
     L0 = -0.4
     L1 = 0.7
     Fr = 0.35
     ALPHA = 1 / Fr
     LAMBDA1 = 1 + ALPHA
     LAMBDA2 = 1 - ALPHA
     T_FINAL = 0.15

     # Domain discretisation
     Nx = 400
     Nt = 220

     # === Analytical solution in characteristic variables ===
     # Forcing function
     s = (lambda x:
          np.where(
               np.abs(x) < 1 / 20,
               np.sin(20 * np.pi * x),
               0
          )
     )
     r = (lambda t:
          np.where(
               np.sin((40 * np.pi * t) + (np.pi / 6)) > 0.5,
               1,
               0
          )
     )
     
     # Create containers for the analytical solution
     domain = Domain(T=T_FINAL, Nx=Nx, Nt=Nt, L_start=L0, L_end=L1)
     x, t = domain.x, domain.t
     p = np.zeros(shape=(len(x), len(t)))
     q = np.zeros(shape=(len(x), len(t)))

     # Compute analytical solution by evaluating the forcing integral along
     # the characteristics
     for i, xn in enumerate(x):
         for j, tn in enumerate(t):
          p[i, j] = 0.5 * forcing_integral(
              xn=xn, tn=tn, char_speed=LAMBDA1, s=s, r=r, N=800
          )
          q[i, j] = -0.5 * forcing_integral(
              xn=xn, tn=tn, char_speed=LAMBDA2, s=s, r=r, N=800
          )
     
     # Recast solution in original variables space
     S = np.array([[1, 1],
                   [1, -1]])
     pq = np.stack((p, q), axis=0)        # shape (2, Nx, Nt)
     uv = S @ pq.reshape(2, -1)           # (2, 2) @ (2, Nx*Nt) → (2, Nx*Nt)
     u, v = uv.reshape(2, len(x), len(t))         # back to (2, Nx, Nt)

     # Save computed analytical solution for future use
     os.makedirs('CE3/results', exist_ok=True)
     filename = f'CE3/results/analytical_sol_Nx{Nx}_Nt{Nt}'
     np.savez(filename, u, v)

     # Save plots to figures to compose animation later
     if savefig:
          fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
          for n, (un, vn) in enumerate(zip(u.T, v.T)):
               save2jpg(u=un, v=vn, x=x, time_step=n, fig=fig, axes=axes)


if __name__ == "__main__":
     main() 
