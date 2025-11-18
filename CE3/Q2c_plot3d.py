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
from plot_util import plot_system_specified_time, plot_system_3d
import os

def run_task(task: str, Nx: int, Nt: int, plot_analytic=True):
     # Problem parameters
     L0 = -0.4
     L1 = 0.7
     Fr = 0.35
     ALPHA = 1 / Fr
     T_FINAL = 0.15

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
     f1 = lambda x, t: np.zeros_like(x)
     f2 = lambda x, t: s(x) * r(t)
     F = [f1, f2]

     # Numerical solution
     domain = Domain(T=T_FINAL, Nx=Nx, Nt=Nt, L_start=L0, L_end=L1)
     init_cond = np.zeros(shape=(len(F), len(domain.x)))
     A = np.array([[1, ALPHA],
                   [ALPHA, 1]])
     solver = SolverSystem(A=A, F=F)
     sols_all_schemes = []
     labels = [scheme.value for scheme in NumericalSchemes]
     for scheme in NumericalSchemes:
          sol_this_scheme = solver.solve_pde(
               domain=domain,
               initial_condition=init_cond,
               num_scheme=scheme,
               CFL_override=True)
          sols_all_schemes.append(sol_this_scheme)

     # Plot solutions
     if task == 'plot_final':
          if plot_analytic:
               # Load analytical solution
               script_dir = os.path.dirname(os.path.abspath(__file__))
               fname_analytic = os.path.join(script_dir, f'analytical_sol_Nx{Nx}_Nt{Nt}.npz')
               sol_analytical= np.load(fname_analytic)
               u, v = sol_analytical['arr_0'], sol_analytical['arr_1']
               sol_analytical = np.stack((u, v), axis=0)
               labels.append('Analytical')
               sols_all_schemes.append(sol_analytical)

          # Plot numerical and analytical solutions          
          plot_system_specified_time(
               domain=domain, plot_time=T_FINAL,
               sols_all_schemes=sols_all_schemes, scheme_labels=labels)
     elif task == 'plot_3d':
          plot_system_3d(
               domain=domain, 
               sols_all_schemes=sols_all_schemes, scheme_names=labels)


if __name__ == "__main__":
     # === Domain discretisation ===
     # Stable selection
     Nx = 400
     Nt = 220
     # # Unstable selection
     # Nx = 400
     # Nt = 208

     # Comment out to select task
     # task = 'plot_final'
     task = 'plot_3d'

     run_task(task, Nx, Nt) 
