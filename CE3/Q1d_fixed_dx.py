# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 3, Part 1-a
#
# Author: Alessio / Tim
# Date: 14 October 2025
# -----------------------------------------------------------------------------
from functools import partial
from solver import Solver, NumericalScheme
from domain import make_domain
import numpy as np
from scipy import signal
from plot_util import plot_multiple_discretisations

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
    # Nt = 85
    Co_vals = [0.1, 0.5, 0.99, 1.0, 1.2]


    # === Boundary conditions ===
    # Create sine/square wave functions of time only u(0,t) = g(t), by taking
    # wave functions w(t; tau) and fixing the period tau
    sine_wave_tau_fixed = partial(sine_wave, tau=TAU)
    square_wave_tau_fixed = partial(square_wave, tau=TAU)
    bc_func = sine_wave_tau_fixed
    bc_plot_title = ' - sine wave BC'
    # bc_func = square_wave_tau_fixed
    # bc_plot_title = ' - square wave BC'

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
    solver = Solver(a=a)
    # domain = Domain(L=D, T=T_FINAL, Nx=Nx, Nt=Nt)

    domains = []
    solutions_numerical_all_domains = []
    for Co in Co_vals:
        domain = make_domain(a=a, L=D, T=T_FINAL, Co=Co, Nx=Nx)
        domains.append(domain)

        # solutions_numerical_all_BCs = []
        # for bc_func in [
        #     sine_wave_tau_fixed,
        #     square_wave_tau_fixed
        # ]:
            # solutions_numerical_thisBC = []
        solutions_numerical_this_domain = []
        # solutions_numerical_sine_bc = []
        for scheme in [
            NumericalScheme.UPWIND,
            NumericalScheme.LAX_FREDRICHS,
            NumericalScheme.LAX_WENDROFF
        ]:
            solutions_numerical_this_domain.append(
                solver.solve_pde(
                    domain=domain,
                    initial_condition=IC_allzero,
                    left_bc=bc_func,
                    num_scheme=scheme
                )
            )
        solutions_numerical_all_domains.append(solutions_numerical_this_domain)
    
    # === Plot analytical and numerical solutions for each BC type ===
    plot_titles = [f'$u(x,{T_FINAL:.15g})$' + bc_plot_title]
    sol_numerical_labels = [scheme.value for scheme in NumericalScheme]
    for (sol_analytical, sols_num, title) in zip(
        solutions_analytical,
        solutions_numerical_all_domains,
        plot_titles
    ):
        plot_multiple_discretisations(
            domains=domains,
            plot_time=T_FINAL,
            sol_analytical=sol_analytical,
            sols_numerical=sols_num,
            sol_numerical_labels=sol_numerical_labels,
            title=title
        )