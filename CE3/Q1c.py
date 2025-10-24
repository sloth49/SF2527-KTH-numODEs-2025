# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 3, Part 1-a
#
# Author: Alessio / Tim
# Date: 14 October 2025
# -----------------------------------------------------------------------------
from functools import partial
from solver import Solver, NumericalScheme
from domain import Domain
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal

def sine_wave(t: float, tau: float):
    return np.sin(2 * np.pi * t / tau)


def square_wave(t: float, tau: float):
    return float(signal.square(2 * np.pi * t / tau))


def plot_time_level_samples(
        x: np.ndarray, t: np.ndarray, u: np.ndarray, Nt: int,
        samples: int = 8
        ) -> None:
    """
    Plot wireframe of solution at different time levels.
    """
    # Downsample results to plot set time levels vs x
    sampling_step = (Nt + 1) // samples
    time_samples = t[::sampling_step]
    solution_samples = u[::sampling_step, :]

    # Colors for each time slice
    cmap = plt.colormaps['viridis']
    line_colours = cmap(np.linspace(0, 1, len(time_samples)))

    # Plot slices
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection='3d')
    for ti, ui, colour in zip(time_samples, solution_samples, line_colours):
        ax.plot(x, np.full_like(a=x, fill_value=ti), ui, '.', c=colour)
    ax.set_xlabel('$x$', fontsize=16, labelpad=10)
    ax.set_ylabel('$t$', fontsize=16, labelpad=10)
    ax.set_zlabel('$u$', fontsize=16, labelpad=10) # type: ignore
    ax.set_zlim(bottom=-2.0, top=2.0) # type: ignore
    ax.view_init(elev=25, azim=-140) # type: ignore
    plt.show()


def plot_at_specified_time(
        x, plot_time,
        sol_analytical, sols_numerical,
        sol_numerical_labels):
    """
    Plots a group of numerical solutions, evaluated at a specified time,
    together with the reference analytical solution evaluated at the same time.

    Parameters:
        x (numpy array, shape (n,)):
            solution support
        plot_time (float):
            specified time for numerical / analytical solution
        sol_analytical (callable):
            function representing the analytical solution        
        sols_numerical (list of numpy arrays):
            list of numerical solutions computed with various schemes, with 
            shape (nTS, n) where nTS = time steps, n = space steps
        sol_numerical_labels (list of strings):
            names of the schemes corresponding to the numerical solutions
    Returns:
        None
    """
    numerical_sols = [num_sol[-1, :] for num_sol in sols_numerical] # extract u(x,plot_time)
    analytical = np.array([sol_analytical(xi, plot_time) for xi in x])
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(x, analytical, label='Analytical solution', linestyle='-')
    for numerical, label in zip(
            numerical_sols,
            sol_numerical_labels):
        plt.plot(x, numerical, label=label,
                 marker='.', markersize=4, linewidth=0)
    plt.grid()
    plt.xlabel('$x$', fontsize=16)
    plt.legend(fontsize=12)
    plt.show()


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
    solver = Solver(a=a)
    domain = Domain(L=D, T=T_FINAL, Nx=Nx, Nt=Nt)
    # Solve with sine wave BC
    solutions_numerical_sine_bc = []
    for scheme in [
        NumericalScheme.UPWIND,
        NumericalScheme.LAX_FREDRICHS,
        NumericalScheme.LAX_WENDROFF]:
        solutions_numerical_sine_bc.append(
            solver.solve_pde(
            domain=domain,
            initial_condition=IC_allzero,
            left_bc=sine_wave_tau_fixed,
            num_scheme=scheme)
        )
    # Solve with square wave BC
    solutions_numerical_square_bc = []
    for scheme in [
        NumericalScheme.UPWIND,
        NumericalScheme.LAX_FREDRICHS,
        NumericalScheme.LAX_WENDROFF]:
        solutions_numerical_square_bc.append(
            solver.solve_pde(
            domain=domain,
            initial_condition=IC_allzero,
            left_bc=square_wave_tau_fixed,
            num_scheme=scheme)
        )
    solutions_numerical_all_BCs = [
        solutions_numerical_sine_bc,
        solutions_numerical_square_bc]
    sol_numerical_labels = [
        NumericalScheme.UPWIND.value,
        NumericalScheme.LAX_FREDRICHS.value,
        NumericalScheme.LAX_WENDROFF.value]
    
    # === Plot analytical and numerical solutions for each BC type ===
    for sol_analytical, sols_num in zip(
            solutions_analytical,
            solutions_numerical_all_BCs):
        plot_at_specified_time(
            x=domain.x, plot_time=T_FINAL,
            sol_analytical=sol_analytical,
            sols_numerical=sols_num,
            sol_numerical_labels=sol_numerical_labels)