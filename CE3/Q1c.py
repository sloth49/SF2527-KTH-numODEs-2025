# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 3, Part 1-a
#
# Author: Alessio / Tim
# Date: 14 October 2025
# -----------------------------------------------------------------------------
from functools import partial
from CE3.solver_system import SolverSystem, NumericalSchemes
from domain import Domain
import numpy as np
from scipy import signal
from plot_util import plot_at_specified_time

def sine_wave(t: float, tau: float):
    return np.sin(2 * np.pi * t / tau)


def square_wave(t: float, tau: float):
    return float(signal.square(2 * np.pi * t / tau))


if __name__ == "__main__":
    # Problem parameters
    TAU = 2.0
    D = 10.0
    a = 2.0
    T_FINAL = 4.0
    Nx = 100
    Nt = 85

    # === Boundary conditions ===
    # Create sine/square wave functions of time only u(0,t) = g(t), by taking
    # wave functions w(t; tau) and fixing the period tau
    sine_wave_tau_fixed = partial(sine_wave, tau=TAU)
    square_wave_tau_fixed = partial(square_wave, tau=TAU)

    # === Analytical solutions ===
    # lambda functions of (x, t) representing the exact PDE solutions
    # u(x,t) = g(t - x/a) for the two given boundary conditions g_sine, g_square
    solutions_analytical = []
    for bc_func in [
        sine_wave_tau_fixed,
        square_wave_tau_fixed
    ]:
        solutions_analytical.append(
            lambda x, time, func=bc_func: func(t=(time -  x / a))
            if x < a * time
            else 0
        )

    # === Initial condition ===
    IC_allzero = np.zeros(shape=Nx+1)

    # === Solve equation numerically ===
    solver = SolverSystem(a=a)
    domain = Domain(L_end=D, T=T_FINAL, Nx=Nx, Nt=Nt)

    sol_numerical_labels = [scheme.value for scheme in NumericalSchemes]
    solutions_numerical_all_BCs = []
    for bc_func in [
        sine_wave_tau_fixed,
        square_wave_tau_fixed
    ]:
        solutions_numerical_thisBC = []
        # solutions_numerical_sine_bc = []
        for scheme in [
            NumericalSchemes.UPWIND,
            NumericalSchemes.LAX_FREDRICHS,
            NumericalSchemes.LAX_WENDROFF
        ]:
            solutions_numerical_thisBC.append(
                solver.solve_pde(
                    domain=domain,
                    initial_condition=IC_allzero,
                    left_bc=bc_func,
                    num_scheme=scheme
                )
            )
        solutions_numerical_all_BCs.append(solutions_numerical_thisBC)
    
    # === Plot analytical and numerical solutions for each BC type ===
    plot_titles = [
        f'$u(x,{T_FINAL:.15g})$ - sine wave BC',
        f'$u(x,{T_FINAL:.15g})$ - square wave BC'
    ]
    for (sol_analytical,
         sols_num,
         title) in zip(
            solutions_analytical,
            solutions_numerical_all_BCs,
            plot_titles
    ):
        plot_at_specified_time(
            domain=domain,
            plot_time=T_FINAL,
            sol_analytical_func=sol_analytical,
            sols_numerical=sols_num,
            sol_numerical_labels=sol_numerical_labels,
            title=title
        )