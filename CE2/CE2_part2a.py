# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 2, Part 2a
#
# Author: Alessio / Tim
# Date: 25 September 2025
# -----------------------------------------------------------------------------

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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


def get_rhs(X: np.ndarray, Y: np.ndarray, T_ext: float,
            h: float, k: float) -> np.ndarray:
    """
    Returns the RHS vector in the discretised linear system

    Parameters:
        X, Y: 2D array of coordinates
        T_ext: external temperature at the bottom boundary
        h: step size
        k: time step size
    """

    rhs = k * f_grid(X, Y).ravel(order='C')   # reshape forcing function as 1D vector

    # Impose the Dirichlet BC at the bottom side of the domain
    nodes_x_direction = X.shape[1]
    rhs[:(nodes_x_direction)] = T_ext * (k / h**2)

    return rhs


# Define the problem parameters
Lx = 12.0
Ly = 5.0
Nx = 60
T_EXT = 25.0
TAU_FINAL = 40.0
DELTA_T = 0.4
X_PROBE, Y_PROBE = 6.0, 2.0

# Domain discretisation
x, y, X, Y, h, Ny = create_grid(Lx, Ly, Nx)

# TODO immplement the time-stepping loop
# Set up linear system and solve
Lp2d = laplacian_2d(Nx=Nx, Ny=Ny, h=h)
L_plus = sp.eye((Nx+1)*(Ny+1), format='csr') + (0.5 * DELTA_T) * Lp2d
L_minus = spla.splu(
    sp.eye((Nx+1)*(Ny+1), format='csr') - (0.5 * DELTA_T) * Lp2d
)
rhs = get_rhs(X=X, Y=Y, T_ext=T_EXT, h=h, k=DELTA_T)
T = L_minus.solve(rhs)
T_grid = T.reshape((Ny+1,Nx+1), order='C')

# # Compute temperature at probe location
# interp = RegularGridInterpolator((y,x), T_grid)
# print('T(x=6, y=2): ', interp((Y_PROBE, X_PROBE)))

# Plot temperature distribution
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(X=X, Y=Y, Z=T_grid, cmap=cm.viridis, linewidth=0, antialiased=False)
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$y$', fontsize=14)
ax.view_init(40, -30)
fig.suptitle('Temperature distribution $T(x,y)', fontsize=16)
plt.tight_layout()
plt.show()