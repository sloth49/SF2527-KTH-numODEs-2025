# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 3, Part 1-a
#
# Author: Alessio / Tim
# Date: 14 October 2025
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from itertools import cycle
from domain import Domain
from math import ceil

def plot_time_level_samples(
        x: np.ndarray, t: np.ndarray, u: np.ndarray, Nt: int,
        samples: int=8
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
        domain: Domain,
        plot_time: float,
        sol_analytical_func: Callable[[float, float], float],
        sols_numerical: list[np.ndarray],
        sol_numerical_labels: list[str],
        title: str) -> None:
    """
    Plots a group of numerical solutions, evaluated at a specified time,
    together with the reference analytical solution evaluated at the same time.

    Parameters:
        domain (Domain):
            Discretised domain over which the numerical solutions are computed
        plot_time (float):
            Specified time for numerical / analytical solution
        sol_analytical (Callable[[float, float], float]):
            Function representing the analytical solution        
        sols_numerical (list(np.ndarray)):
            List of arrays, with shape (nTS, n) where nTS = time steps,
            n = space steps, representing the numerical solutions computed with
            various schemes
        sol_numerical_labels (list(str)):
            Names of the schemes corresponding to the numerical solutions

    Returns:
        None
    """
    # Extract numerical solutions values u(x,t=plot_time)
    x, t = domain.x, domain.t
    plot_time_index = np.argmin(np.abs(plot_time - t))
    numerical_sols = [sol[plot_time_index, :] for sol in sols_numerical]

    # Compute analytical solution
    sol_analytical = np.array([sol_analytical_func(xi, plot_time) for xi in x])
    
    # Plot results
    fig = plt.figure(figsize=(8,6))
    markers = cycle(('o', 'v', 's', '*', 'D', 'X', '^'))
    plt.plot(x, sol_analytical, label='Analytical solution', linestyle='-')
    for numerical, label in zip(
            numerical_sols,
            sol_numerical_labels):
        plt.plot(x, numerical, label=label,
                 marker=next(markers), markersize=4, linewidth=0)
    plt.grid()
    plt.xlabel('$x$', fontsize=14)
    plt.legend(fontsize=12)
    plt.title(title, fontsize=16, pad=10)
    plt.show()


def plot_multiple_discretisations(
        
        domains: list[Domain],
        plot_time: float,
        sol_analytical_func: Callable[[float, float], float],
        sols_numerical: list[np.ndarray],
        sol_numerical_labels: list[str],
        title: str,
        plot_columns: int=2) -> None:
    """
    Plots multiple subplots, one for each specified domain discretisation.

    Parameters:
        domains (list[Domain]):
            list of discretised domains
        plot_time (float):
            Specified time for numerical / analytical solution
        sol_analytical_func (Callable[[float, float], float]):
            Function representing the analytical solution        
        sols_numerical (list(list(np.ndarray))):
            List of lists of arrays, with shape (nTS, n) where nTS = time steps,
            n = space steps, representing the numerical solutions computed with
            various schemes and various values of the parameter
        sol_numerical_labels (list(str)):
            Names of the schemes corresponding to the numerical solutions
        title (str):
            Plot title

    Returns:
        None
    """
    # Min. number of rows needed to fit all plots
    plot_rows = ceil(len(domains) / plot_columns) 

    fig, axes = plt.subplots(nrows=plot_rows, ncols=plot_columns)
    markers = cycle(('o', 'v', 's', '*', 'D', 'X', '^'))

    # x = np.linspace(0, 5)
    # y = np.sin(x)
    # for ax in axes.ravel():
    #     ax.plot(x, y)
    for domain, ax in zip(domains, axes.ravel()):
        x, t = domain.x, domain.t

        # Analytical solution
        sol_analytical = np.array([sol_analytical_func(xi, plot_time) for xi in x])

        # Extract numerical solutions values u(x,t=plot_time)
        plot_time_index = np.argmin(np.abs(plot_time - t))
        numerical_sols = [sol[plot_time_index, :] for sol in sols_numerical]

        # Plot results
        ax.plot(x, sol_analytical, label='Analytical solution', linestyle='-')
        for numerical_sol, label in zip(numerical_sols,sol_numerical_labels):
            ax.plot(
                x, numerical_sol,
                label=label, marker=next(markers), markersize=4, linewidth=0
            )
        ax.grid()
        ax.set_xlabel('$x$', fontsize=14)
        ax.set_title(f'dt: {domain.dt}')
        ax.legend(fontsize=12)
    fig.suptitle(title, fontsize=16)
    plt.show()