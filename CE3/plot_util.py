# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 3, Part 1-a
#
# Author: Alessio / Tim
# Date: 14 October 2025
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable
from itertools import cycle
from domain import Domain
from math import ceil
import plotly.graph_objects as go


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
                 marker=next(markers), markersize=3, linewidth=0.4)
    plt.grid()
    plt.xlabel('$x$', fontsize=14)
    plt.legend(fontsize=12)
    plt.title(title, fontsize=16, pad=10)
    plt.show()


def plot_multiple_discretisations(
        
        domains: list[Domain],
        Co_vals: list[float],
        plot_time: float,
        sol_analytical_func: Callable[[float, float], float],
        sols_num_all_domains: list[list[np.ndarray]],
        sol_num_labels: list[str],
        title: str,
        plot_columns: int=2) -> None:
    """
    Plots multiple subplots, one for each specified domain discretisation.

    Parameters:
        domains (list[Domain]):
            List of discretised domains
        Co_vals (list[float]):
            List of Courant numbers resulting from the combination of advection
            velocity and domain discretisation
        plot_time (float):
            Specified time for numerical / analytical solution
        sol_analytical_func (Callable[[float, float], float]):
            Function representing the analytical solution        
        sols_num_all_domains (list(list(np.ndarray))):
            List of lists of arrays, with shape (nTS, n) where nTS = time steps,
            n = space steps, representing the numerical solutions computed with
            various schemes and various values of the parameter
        sol_num_labels (list(str)):
            Names of the schemes corresponding to the numerical solutions
        title (str):
            Plot title

    Returns:
        None
    """
    # Min. number of rows needed to fit all plots
    plot_rows = ceil(len(domains) / plot_columns) 

    fig, axes = plt.subplots(nrows=plot_rows, ncols=plot_columns)
    markers_sequence = ('o', 'v', '.', '*', 'D', 'X', '^')

    for domain, Co, sols_num_this_domain, ax in zip(
        domains,
        Co_vals,
        sols_num_all_domains,
        axes.ravel()
    ):
        markers = cycle(markers_sequence)
        x, t = domain.x, domain.t

        # Analytical solution
        sol_analytical = np.array([sol_analytical_func(xi, plot_time) for xi in x])

        # Extract numerical solutions values u(x,t=plot_time)
        plot_time_index = np.argmin(np.abs(plot_time - t))
        sols_num = [sol[plot_time_index, :] for sol in sols_num_this_domain]

        # Plot results
        ax.plot(x, sol_analytical, label='Analytical solution', linestyle='-')
        for sol_num, label in zip(sols_num, sol_num_labels):
            ax.plot(
                x, sol_num,
                label=label, marker=next(markers), markersize=3, linewidth=0.4
            )
        ax.grid()
        ax.set_xlabel('$x$', fontsize=10)
        ax.set_title(f'dt: {domain.dt:.3f}, Co: {Co:.3f}, Nx: {domain.Nx}')
        if ax == axes[0, 0]:
            ax.legend(fontsize=12)

    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(hspace=0.4)
    plt.show()

    
def save2jpg(u, v, x, time_step, fig, axes):
    [ax.clear() for ax in axes]
    ax_u, ax_v = axes

    ax_u.plot(x, u, '-')
    ax_u.set_title('$u$', fontsize=16)
    ax_u.set_ylim(-0.01, 0.01)
    ax_u.set_xlabel('$x$', fontsize=14)
    ax_u.grid(alpha=0.7)

    ax_v.plot(x, v, '-')
    ax_v.set_title('$v$', fontsize=16)
    ax_v.set_ylim(-0.01, 0.01)
    ax_v.set_xlabel('$x$', fontsize=14)
    ax_v.grid(alpha=0.7)

    plt.tight_layout()
    plt.savefig(
        f'CE3/figures/CE3_part2b_analytical_{time_step}.jpg',
        format='jpg',
        dpi=300)


def plot_system_specified_time(
        domain: Domain,
        plot_time: float,
        sols_all_schemes: list[np.ndarray],
        scheme_labels: list[str]
    ) -> None:
    # Extract numerical solution at specified time, list of one item per scheme 
    x, t = domain.x, domain.t
    plot_time_index = np.argmin(np.abs(plot_time - t))
    sols_all_schemes_at_t = [sol[:, :, plot_time_index] for sol in sols_all_schemes]

    # Plot results
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,6))
    var_names = ['$u$', '$v$']
    
    for i_var, ax in enumerate(axes):
        markers = cycle(('v', '^', '.'))
        marker_sizes = cycle((3, 3, 4))
        for sol_this_scheme_at_t, scheme_label in zip(sols_all_schemes_at_t, scheme_labels):
            marker=next(markers)
            var = sol_this_scheme_at_t[i_var]
            ax.plot(x, var, label=scheme_label,
                    marker=next(markers), markersize=next(marker_sizes), linewidth=0.3)
    for ax, var_name in zip(axes, var_names):
        ax.grid()
        ax.set_ylabel(var_name, fontsize=18, rotation=0, labelpad=15)
        ax.set_ylim(bottom=0.008, top=-0.008)

    axes[0].legend(fontsize=12)
    axes[1].set_xlabel('$x$', fontsize=18, labelpad=15)
    fig.suptitle(f'Numerical solution at time t={plot_time}', fontsize=14)
    plt.show()


def plot_system_3d(
        domain: Domain,
        sols_all_schemes: list[np.ndarray],
        scheme_names: list[str]
    ) -> None:
    """
    Generates interactive 3d plots in HTML format, which can be rotated / zoomed in
    using your mouse.

    Parameters:
        domain (Domain):
            The integration domain of the PDE
        sols_all_schemes (list[np.ndarray]):
            A list of Numpy arrays representing the solutions for different
            numerical schemes / analytical. Item shape: (No of vars, Nx+1, Nt+1)
        scheme_names (list[str]):
            List of names for each scheme provided in sols_all_schemes
    """
    X, Y = domain.get_meshgrid()

    for sol_this_scheme, scheme_name in zip(sols_all_schemes, scheme_names):
        U = sol_this_scheme[0]

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=U,
                    colorscale='Viridis',
                    opacity=0.8,        # semi-transparent
                    showscale=True
                )
            ]
        )

        fig.add_trace(
            go.Scatter3d(
                x=X.flatten(),
                y=Y.flatten(),
                z=U.flatten(),
                mode='markers',
                marker=dict(size=3, color=U.flatten(), colorscale='Viridis'),
                name='grid points'
            )
        )


        fig.update_layout(
            title=scheme_name,
            scene=dict(
                xaxis_title='x',
                yaxis_title='t',
                zaxis_title='u(x,t)',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        fig.show()
