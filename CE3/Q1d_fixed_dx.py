# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 3, Part 1-a
#
# Author: Alessio / Tim
# Date: 14 October 2025
# -----------------------------------------------------------------------------
from functools import partial
from solver import Solver, NumericalSchemes
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
    Co_vals = [0.1, 0.3, 0.5, 0.7, 0.95, 1.0]
    plot_time = T_FINAL
    # plot_time = 0.5


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
    # lambda function of (x, t) representing the exact PDE solution
    # u(x,t) = g(t - x/a) for the given boundary condition (g_sine or g_square)
    sol_analytical_func = (lambda x, time, func=bc_func:
                           func(t = (time -  x / a))
                           if x < a * time
                           else 0)

    # === Initial condition ===
    IC_allzero = np.zeros(shape=Nx+1)

    # === Solve equation numerically ===
    solver = Solver(a=a)
    domains = []
    solutions_numerical_all_domains = []
    for Co in Co_vals:
        domain = make_domain(a=a, L=D, T=T_FINAL, Co=Co, Nx=Nx)
        domains.append(domain)
        solutions_numerical_this_domain = []
        for scheme in NumericalSchemes:
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
    plot_titles = [f'$u(x,{T_FINAL:.2f})$' + bc_plot_title]
    sol_numerical_labels = [scheme.value for scheme in NumericalSchemes]
    for title in plot_titles:
        plot_multiple_discretisations(
            domains=domains,
            Co_vals=Co_vals,
            plot_time=plot_time,
            sol_analytical_func=sol_analytical_func,
            sols_num_all_domains=solutions_numerical_all_domains,
            sol_num_labels=sol_numerical_labels,
            title=title
        )