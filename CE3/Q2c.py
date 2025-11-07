# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 3, Part 2-c
#
# Author: Alessio / Tim
# Date: 3 November 2025
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from solver_system import SolverSystem, NumericalSchemes
from domain import Domain

def main():
     # Problem parameters
     L0 = -0.4
     L1 = 0.7
     Fr = 0.35
     ALPHA = 1 / Fr
     T_FINAL = 0.15

     # Forcing function
     s = (lambda csi:
          np.where(
               np.abs(csi) < 1 / 20,
               np.sin(20 * np.pi * csi),
               0
          )
     )
     r = (lambda tau:
          np.where(
               np.sin((40 * np.pi * tau) + (np.pi / 6)) > 0.5,
               1,
               0
          )
     )
     f1 = lambda x, t: np.zeros(shape=(len(x), len(t)))
     f2 = lambda x, t: s(x) * r(t)
     F = [f1, f2]

     # Domain discretisation
     Nx = 100
     Nt = 100
     domain = Domain(T=T_FINAL, Nx=Nx, Nt=Nt, L_start=L0, L_end=L1)

     # PDE setup
     init_cond = np.zeros(shape=(len(F), len(domain.x)))
     A = np.array([[1, ALPHA],
                   [ALPHA, 1]])
     solver = SolverSystem(A=A, F=F)
     soln = solver.solve_pde(
          domain=domain,
          initial_condition=init_cond,
          num_scheme=NumericalSchemes.UPWIND)


if __name__ == "__main__":
     main() 
