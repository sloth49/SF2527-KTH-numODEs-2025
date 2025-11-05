# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 3, Part 2-c
#
# Author: Alessio / Tim
# Date: 3 November 2025
# -----------------------------------------------------------------------------

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from solver import Solver, NumericalSchemes
from domain import Domain
from plot_util import save2jpg

def main():
     # Problem parameters
     L0 = -0.4
     L1 = 0.7
     Fr = 0.35
     ALPHA = 1 / Fr
     LAMBDA1 = 1 + ALPHA
     LAMBDA2 = 1 - ALPHA
     T_FINAL = 0.15

     # Domain discretisation
     Nx = 100
     Nt = 50

     # Characteristic variables
     S = np.array([[1, 1],
                   [1, -1]])
     
     s = (lambda csi:
          np.sin(20 * np.pi * csi)
          if np.abs(csi) < 1 / 20
          else 0)

     r = (lambda tau:
          1
          if np.sin((40 * np.pi * tau) + (np.pi / 6)) > 0.5
          else 0)
     
     def forcing_along_char(tau, x, t, char_speed):
          csi = x - char_speed * (t - tau)
          return s(csi) * r(tau)
     

     # Compute analytical solution
     domain = Domain(T=T_FINAL, Nx=Nx, Nt=Nt, L_start=L0, L_end=L1)
     x, t = domain.x, domain.t
     p = np.zeros(shape=(len(x), len(t)))
     q = np.zeros(shape=(len(x), len(t)))

     for i, xn in enumerate(x):
         for j, tn in enumerate(t):
          p[i, j] = 0.5 * quad(
               func=forcing_along_char,
               a=0, b=tn, limit=200, args=(xn, tn, LAMBDA1)
          )[0]
          q[i, j] = -0.5 * quad(
               func=forcing_along_char,
               a=0, b=tn, limit=200, args=(xn, tn, LAMBDA2)
          )[0]
     
     pq = np.stack((p, q), axis=0)        # shape (2, Nx, Nt)
     uv = S @ pq.reshape(2, -1)           # (2, 2) @ (2, Nx*Nt) → (2, Nx*Nt)
     u, v = uv.reshape(2, len(x), len(t))         # back to (2, Nx, Nt)

     # Plot analytical solution
     fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
     for n, (un, vn) in enumerate(zip(u.T, v.T)):
         save2jpg(u=un, v=vn, x=x, time_step=n, fig=fig, axes=axes)

if __name__ == "__main__":
     main() 
