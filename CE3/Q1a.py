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
from scipy import signal

def sine_wave(t: float, tau: float):
    return np.sin(2 * np.pi * t / tau)


def square_wave(t: float, tau: float):
    return signal.square(2 * np.pi * t / tau)


def main(boundary_condition, plot_title):
    """
    Plot the analytical solution ton CE3 part 1, for square wave
    and sine wave boundary conditions
    """
    # === PDE to solve ===
    # advection equation with constant adv. speed:
    #    u_t + C u_x = 0

    # === Problem set up ===
    D = 10.0      # spatial domain length
    T_END = 4.0  # time domain length
    TAU = 2.0    # BC sinewave period
    Nx = 100      # space domain intervals (Nx + 1 nodes)
    Nt = 50     # time domain intervals (Nt + 1 nodes)
    a = 2.0      # advection speed

    # Discretise domain
    x = np.linspace(start=0.0, stop=D, num=Nx+1)
    t = np.linspace(start=0.0, stop=T_END, num=Nt+1)
    X, T = np.meshgrid(x, t)

    # calculate analytical solution
    u = np.zeros_like(X)
    u = np.where(
        (X < a * T),
        boundary_condition(t=(T - X / a), tau=TAU),
        0
    ) # type: ignore

    # === Plot results ===
    
    # Downsample results to plot solution at fixed time levels
    samples_number = 8
    sampling_step = (Nt + 1) // samples_number
    time_samples = t[::sampling_step]
    solution_samples = u[::sampling_step, :]

    # Plot wireframes
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection='3d')
    # Colors for each time slice
    cmap = plt.colormaps['viridis']
    line_colours = cmap(np.linspace(0, 1, len(time_samples)))
    for ti, ui, colour in zip(time_samples, solution_samples, line_colours):
        ax.plot(x, np.full_like(a=x, fill_value=ti), ui, c=colour)
    ax.set_xlabel('$x$', fontsize=16, labelpad=10)
    ax.set_ylabel('$t$', fontsize=16, labelpad=10)
    ax.set_zlabel('$u$', fontsize=16, labelpad=10) # type: ignore
    ax.set_zlim(bottom=-2.0, top=2.0) # type: ignore
    ax.view_init(elev=25, azim=-140) # type: ignore
    ax.set_title(plot_title, fontsize=16)
    plt.show()


if __name__ == "__main__":
    for bc_func, plot_title in zip(
        [square_wave, sine_wave],
        ['Analytical solution - square wave B.C.', 'Analytical solution - sine wave B.C.']
    ):
        main(bc_func, plot_title)

