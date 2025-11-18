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


def main(func, Co_vals, Nx_vals, task_title):
    # Problem parameters
    TAU = 2.0
    D = 10.0
    a = 2.0
    T_FINAL = 4.0
    plot_time = T_FINAL

    # === Boundary conditions ===
    # Create sine/square wave functions of time only u(0,t) = g(t), by taking
    # wave functions w(t; tau) and fixing the period tau
    bc_func_tau_fixed = partial(func, tau=TAU)
    # bc_func = sine_wave_tau_fixed
    if func == sine_wave:
        bc_plot_title = 'sine wave BC'
    elif func == square_wave:
        bc_plot_title = 'square wave BC'

    # === Analytical solutions ===
    # lambda function of (x, t) representing the exact PDE solution
    # u(x,t) = g(t - x/a) for the given boundary condition (g_sine or g_square)
    sol_analytical_func = (lambda x, time, func=bc_func_tau_fixed:
                           func(t = (time -  x / a))
                           if x < a * time
                           else 0)

    # === Solve equation numerically ===
    solver = Solver(a=a)
    domains = []
    solutions_numerical_all_domains = []

    for Co, Nx in zip(Co_vals, Nx_vals):
        # Create domain with specified parameters
        domain = make_domain(a=a, L_end=D, T=T_FINAL, Co=Co, Nx=Nx)
        domains.append(domain)

        # Initial condition
        IC_allzero = np.zeros(shape=Nx+1)

        # Compute numerical solutions
        solutions_numerical_this_domain = []
        for scheme in NumericalSchemes:
            solutions_numerical_this_domain.append(
                solver.solve_pde(
                    domain=domain,
                    initial_condition=IC_allzero,
                    left_bc=bc_func_tau_fixed,
                    num_scheme=scheme,
                    CFL_override=True
                )
            )
        solutions_numerical_all_domains.append(solutions_numerical_this_domain)
    
    # === Plot analytical and numerical solutions for each BC type ===
    plot_title = f'$u(x,{T_FINAL:.2f})$ - ' + bc_plot_title + task_title # type: ignore
    sol_numerical_labels = [scheme.value for scheme in NumericalSchemes]
    plot_multiple_discretisations(
        domains=domains,
        Co_vals=Co_vals,
        plot_time=plot_time,
        sol_analytical_func=sol_analytical_func,
        sols_num_all_domains=solutions_numerical_all_domains,
        sol_num_labels=sol_numerical_labels,
        title=plot_title
    )


if __name__ == "__main__":

    # === Fixed Nx, varying Courant number ===
    Nx_fixed = 100
    Co_vals = [0.3, 0.5, 0.7, 0.95, 1.0, 1.001]
    Nx_vals = [Nx_fixed for Co in Co_vals]
    task_title = ' - varying Co'

    # # === Fixed Courant No, varying Nx ===
    # Co_fixed = 0.9
    # Nx_vals = [80, 100, 150, 200, 250, 300]
    # Co_vals = [Co_fixed for Co in Nx_vals]
    # task_title = ' - varying $N_x$'

    for bc in [sine_wave, square_wave]:
        main(bc, Co_vals, Nx_vals, task_title)