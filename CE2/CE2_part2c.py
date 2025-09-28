# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 2, Part 2c
#
# Author: Alessio / Tim
# Date: 25 September 2025
# -----------------------------------------------------------------------------

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


def create_grid(Lx: float, Ly: float, Nx: int):
    """
    Creates a grid of coordinates in a rectangular domain
    No of nodes in (x, y) direction: (N+1, M+1)
    First and last nodes are placed on the boundary of the domain
    The step size is taken as equal in each direction, (dx = dy = h)

    Parameters
        Lx: length in x direction
        Ly: length in y direction
        N: number of division in the x direction

    Returns:
        X: array of x coordinates
        Y: array of y coordinates
        h: step size
        M: No of divisions in y direction
    """
    x = np.linspace(start=0, stop=Lx, num=Nx+1)
    h = x[1] - x[0]

    Ny = int(Ly / h)
    y = np.linspace(start=0, stop=Ly, num=Ny+1)

    X, Y = np.meshgrid(x, y)

    return x, y, X, Y, h, Ny


def Sn_DN(n: int, h: float) -> sp.csr_matrix:
    """
    Finite-difference derivative matrix
    approximating a one-dimensional 2nd spatial derivative, with Dirichlet
    BC at the start, and homogeneous Neumann at the end of the domain.
    To be used in the given equation:

        u_t = (u_xx + u_yy) + f

    Shape:
        (n+1) x (n+1) for n intervals
        It includes the points lying on the boundaries

    Boundary conditions:
      Start point: Dirichlet (T = Text )
                   Note that this makes the node to the right of the boundary
                   note behave like a normal interior point
      End point: Neumann homogeneous

    For n=4, for example: 

               |  1    0                 |
               |  1   -2    1            |
    h^2 * Sn = |       1   -2    1       |
               |            1   -2    1  |
               |                 2   -2  |
    """

    diag_main = np.full(shape=n+1, fill_value=-2)
    diag_main[0] = 1    # Dirichlet BC

    diag_lower = np.full(shape=n, fill_value=1)
    diag_lower[-1] = 2

    diag_upper = np.full(shape=n, fill_value=1)
    diag_upper[0] = 0

    A = sp.diags(
        diagonals=[diag_lower, diag_main, diag_upper],
        offsets=[-1, 0, 1], shape=(n+1, n+1), format='csr')
    
    return A / (h**2)


def Sn_NN(n: int, h: float) -> sp.csr_matrix:
    """
    Finite-difference derivative matrix
    approximating a one-dimensional 2nd spatial derivative, with homogeneous
    Neumann BC at both ends of the domain. To be used in the given equation:

        u_t = (u_xx + u_yy) + f

    Shape:
        (n+1) x (n+1) for n intervals:
        It includes the points lying on the boundaries

    For n=4, for example:

               |  -2    2                |
               |   1   -2    1           |
    h^2 * Sn = |        1   -2    1      |
               |            1   -2    1  |
               |                 2   -2  |
    """

    diag_main = np.full(shape=n+1, fill_value=-2)

    diag_lower = np.full(shape=n, fill_value=1)
    diag_lower[-1] = 2

    diag_upper = np.full(shape=n, fill_value=1)
    diag_upper[0] = 2

    A = sp.diags(
        diagonals=[diag_lower, diag_main, diag_upper],
        offsets=[-1, 0, 1], shape=(n+1, n+1), format='csr')
    
    return A / (h**2)


def laplacian_2d(Nx: int, Ny: int, h: float) -> sp.csr_matrix:
    """
    Implements formula on slides #7 page 22
    """
    S_Nx = Sn_NN(n=Nx, h=h)
    S_Ny = Sn_DN(n=Ny, h=h)
    I_Ny = sp.eye(Ny+1, format="csr")
    I_Nx = sp.eye(Nx+1, format="csr")

    return (sp.kron(I_Ny, S_Nx, format="csr") + sp.kron(S_Ny, I_Nx, format="csr"))


def f_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Forcing function f(x,y)

    Parameters:
        x, y: 2D array of coordinates
    
    Returns:
        2D array of values, shape as input coordinates
    """
    return 100 * np.exp(-0.5 * (x - 4)**2 - 4 * (y - 1)**2)


def get_Lplus_Lminus(Nx, Ny, h, dt):
    """
    Returns the matrices L_plus and L_minus for the Crank-Nicolson scheme
    """
    # Laplacian term discretisation
    Lp2d = laplacian_2d(Nx=Nx, Ny=Ny, h=h)
    I_NxNy = sp.eye((Nx+1)*(Ny+1), format='csr')
    L_plus = I_NxNy + (0.5 * dt) * Lp2d
    L_minus = I_NxNy - (0.5 * dt) * Lp2d

    # Impose Dirichlet BC at bottom boundary
    L_minus.tolil()     # convert to LIL format to make value assignments easier
    L_minus[:(Nx+1),:] = 0.0    # set rows corresponding to bottom side all to zero
    diag = L_minus.diagonal()
    diag[:(Nx+1)] = 1.0  # Set the diagonal entries to 1 to match the RHS modification
    L_minus.setdiag(diag)
    L_minus = L_minus.tocsr()   # convert back to CSR format for efficiency
    L_minus = spla.splu(L_minus)    # LU factorization for more efficiency in time loop

    return L_plus, L_minus


# Define the problem parameters
Lx = 12.0
Ly = 5.0
Nx = 120
T_EXT = 25.0
TAU_FINAL = 40.0
DELTA_T = 0.25
X_PROBE, Y_PROBE = 6.0, 2.0
T_PROBE_CE1 =  47.224998

# Spatial domain discretisation
x, y, X, Y, h, Ny = create_grid(Lx=Lx, Ly=Ly, Nx=Nx)

# create array of time steps
time_steps = int(TAU_FINAL / DELTA_T)
t = np.linspace(start=0, stop=TAU_FINAL, num=time_steps+1) # include initial time t=0 to be able to plot initial condition
dt = t[1] - t[0]  # time step size

L_plus, L_minus = get_Lplus_Lminus(Nx=Nx, Ny=Ny, h=h, dt=dt)

# Calculate time-indepent part of RHS (dt * f)
forcing_term = dt * f_grid(x=X, y=Y).ravel(order='C')

# Time integration (Crank-Nicolson)
u0 = np.full(shape=(Ny+1, Nx+1), fill_value=T_EXT).ravel(order='C')  # Initial condition (block at uniform temperature Text)
u = u0.copy()   # Initialise the solution to the initial condition
rhs = L_plus @ u0 + forcing_term    # Initialise right-hand side
rhs[:(Nx+1)] = T_EXT  # Impose Dirichlet BC at bottom boundary

# Time marching loop and store results for plotting
probe_vals = [T_EXT]
for n in range(1, time_steps+1):
    u = L_minus.solve(rhs)
    u_grid = u.reshape((Ny+1,Nx+1), order='C')
    interp = RegularGridInterpolator((y,x), u_grid)
    probe_val = interp((Y_PROBE, X_PROBE))
    probe_vals.append(probe_val)
    rhs = L_plus @ u + forcing_term
    rhs[:(Nx+1)] = T_EXT  # re-impose Dirichlet BC at bottom boundary

print(f'Temperature at probe location (x={X_PROBE}, y={Y_PROBE})',
      f'at final time ({TAU_FINAL}): {probe_vals[-1]:.6f}')

# plot temperature at probe location
plt.figure(figsize=(8,5))
plt.plot(t, probe_vals, '-o')
plt.xlabel('$\\tau$', fontsize=14)
plt.title('$T(x=6,y=2,\\tau)$', fontsize=16)
plt.axhline(y=T_PROBE_CE1, color='r', linestyle='--', label=f'CE1 value ({T_PROBE_CE1:.4f})')
plt.legend(fontsize=12)
plt.grid()
plt.show()