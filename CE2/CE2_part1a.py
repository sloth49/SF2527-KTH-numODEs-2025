# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 2, Part 1-a
#
# Author: Alessio / Tim
# Date: 23 September 2025
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
import math


def assemble_matrix(dx: float, diffusion_coeff: float) -> sp.csc_matrix:
    """
    Assembles the finite difference matrix for the 1D heat equation
    with Dirichlet boundary conditions.
    """    
    diag_off = np.full(shape=Nx-2, fill_value=1)
    diag_main = np.full(shape=Nx-1, fill_value=-2)
    diag_main[-1] = -1

    A = sp.diags(
        diagonals=[diag_off, diag_main, diag_off],
        offsets=[-1, 0, 1], format='csc')
    A *= (diffusion_coeff / dx**2)

    return A


def alpha(time: float, end_forcing_time: float) -> float:
    """
    Boundary condition function at x=0.
    """
    if time<= end_forcing_time:
        return math.sin(math.pi * (time / end_forcing_time))
    else:
        return 0.0


def integrate_euler_explicit(
        u_initial: np.ndarray,
        space_domain: np.ndarray, time_domain: np.ndarray,
        dx: float, dt: float,
        diffusion_coeff: float, end_forcing: float) -> np.ndarray:
    """
    Integrates the 1D heat equation using the explicit Euler method in time,
    with Dirichlet boundary condition at x=0, Neumann at x=L:
                ut = d * uxx
                u(0,t) = alpha(t),   du/dx(L,t) = 0
    Returns:
        array of shape (N + 1, num_steps + 1) where each row corresponds
        to the solution at a given time step, starting from t=0.
    """
    nodes, time_steps = space_domain.shape[0], time_domain.shape[0] # node: Nx+1, time_steps: Nt+1
    end_time = time_domain[-1]

    A = assemble_matrix(dx=dx, diffusion_coeff=diffusion_coeff)

    # Initialise b vector
    b = np.zeros(shape=nodes-2)
    alpha_tn = alpha(time=0.0, end_forcing_time=end_forcing)
    b[0] = (diffusion_coeff / dx**2) * alpha_tn

    u_grid = np.zeros(shape=(time_steps, nodes))    # solution container
    u_grid[0, :] = u_initial                     # set initial condition

    # begin time-stepping
    u_inner = u_initial[1:-1] 
    for n in range(1, time_steps):

        u_inner_new = u_inner + dt * (A @ u_inner + b)
        u_grid[n, 1:-1] = u_inner_new   # update inner nodes
        u_grid[n, 0] = alpha_tn       # update the BC at x=0
        u_grid[n, -1] = u_grid[n, -2]  # update the BC at x=L (Neumann)
        
        # update for next iteration
        u_inner = u_inner_new
        tn = time_domain[n]
        alpha_tn = alpha(time=tn, end_forcing_time=end_forcing)
        b[0] = (diffusion_coeff / dx**2) * alpha_tn

    return  u_grid


def discretise_domain(
        start_position: float, end_position: float, spatial_intervals:int,
        T_final: float, time_steps: float) -> np.ndarray:
    """
    Discretises the spatial and temporal domains.
    Returns:        
        x and t: arrays of shape (spatial_intervals+1,) and (time_steps+1,)
        dx and dt: floats
    """
    x = np.linspace(start=start_position, stop=end_position, num=spatial_intervals+1)
    dx = x[1] - x[0]

    t = np.linspace(start=0, stop=T_final, num=time_steps+1)
    dt = t[1] - t[0]

    return x, t, dx, dt


# Define parameters

L = 1.0                     # length of the rod
T_FINAL = 2.0               # final time
t_snapshot = 1.1
Nx = 100                  # spatial intervals on x axis - so Nx+1 nodes
Nt = 14000                # time steps - so Nt+1 time levels, starting from t=0 (initial condition)
a = 1.2                     # boundary condition parameter
d = 0.35                    # diffusion coefficient
u0 = np.zeros(shape=Nx+1)   # initial condition for all x including boundaries

#Discretization
x, t, dx, dt = discretise_domain(
    start_position=0.0, end_position=L, spatial_intervals=Nx,
    T_final=T_FINAL, time_steps=Nt)

# Stability check
Co = (d * dt) / dx**2
if (Co > 0.5):
    raise ValueError(f"CFL Condition not met: Co = (d*dt)/(dx^2) = {Co:.4f} > 0.5")
# if False:
#     pass
else:
    print(
        "Solving PDE with Explicit Euler method in time","\n"
        f"Courant No: (d*dt)/(dx^2) = {Co:.4f} <= stability limit (0.5)")

u_grid = integrate_euler_explicit(
    u_initial=u0,
    space_domain=x, time_domain=t,
    dx=dx, dt=dt, diffusion_coeff=d, end_forcing=a)

# Plotting the 2D temperature distribution
X, TAU = np.meshgrid(x, t, indexing='xy')
plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, TAU, u_grid, cmap='viridis')
ax.set_xlabel('x')  
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
ax.set_title('3D surface plot of u(x,t)')
plt.show()

# Plot the distribution at t_snapshot
interp_func = RegularGridInterpolator((t, x), u_grid)
u_snapshot = interp_func((t_snapshot, x))
plt.figure()
plt.plot(x, u_snapshot, label=f't={t_snapshot:.1f}s')
plt.xlabel('$x$') 
plt.ylabel('$u(x,\\tau_1)$')
plt.title('Distribution of u along the rod at t=1.1s')
plt.legend()
plt.grid()
plt.show()

# Plot the temperature evolution at left end (x=0) and right end (x=L)
plt.figure()
plt.plot(t, u_grid[:, 0], label='$u(x=0,\\tau)$')
plt.plot(t, u_grid[:, -1], label='$u(x=L,\\tau)$')
plt.xlabel('$\\tau$')
plt.grid()
plt.title('Temperature evolution at the ends of the rod')
plt.legend()
plt.show()

