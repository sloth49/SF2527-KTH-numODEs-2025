# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 1, Part 4d
#
# Author: Alessio / Tim
# Date: 26 August 2025
# -----------------------------------------------------------------------------

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def create_grid(Lx: float, Ly: float, N: int):
    """
    Creates a grid of coordinates in a rectangular domain
    No of nodes in (x, y) direction: (N+1, M+1)
    First and last nodes are placed on the boundary of the domain
    The step size is taken as equal in each direction

    Parameters
        Lx: length in x direction
        Ly: length in y direction
        N: number of division in the x direction

    Returns:
        X: array of x coordinates
        Y: array of y coordinates
        h: step size
        M: No of steps in y direction (ca)
    """
    x = np.linspace(start=0, stop=Lx, num=N+1)
    h = x[1] - x[0]

    M = int(Ly / h)
    y = np.linspace(start=0, stop=Ly, num=M+1)

    X, Y = np.meshgrid(x, y)

    return x, y, X, Y, h, M


def Sn_DN(n: int, h: float) -> sp.csr_matrix:
    """
    One-dimensional finite-difference derivative matrix 
    of size (n+1) x (n+1) solving:
        -u'' = f

    Boundary conditions:
      Start point: Dirichlet (T0 = alpha )
                   (note that this makes the node next to the left
                    boundary behave like a normal interior point)
      End point: Neumann homogeneous 

    For n=4, for example: 

               |  1    0                 |
               |  -1   2   -1            |
    h^2 * Sn = |      -1    2   -1       |
               |           -1    2   -1  |
               |                -2    2  |
    """

    diag_main = np.full(shape=n+1, fill_value=2)
    diag_main[0] = 1    # Dirichlet BC

    diag_lower = np.full(shape=n, fill_value=-1)
    diag_lower[-1] = -2

    diag_upper = np.full(shape=n, fill_value=-1)
    diag_upper[0] = 0

    A = sp.diags(
        diagonals=[diag_lower, diag_main, diag_upper],
        offsets=[-1, 0, 1], shape=(n+1, n+1), format='csr')
    
    return A / (h**2)


def Sn_NN(n: int, h: float) -> sp.csr_matrix:
    """
    One-dimensional finite-difference derivative matrix 
    of size (n+1) x (n+1) solving:
        -u'' = f

    Boundary conditions:
      Start point: Neumann homogeneous
      End point: Neumann homogeneous 

    For n=4, for example:

               |  2   -2                 |
               | -1    2   -1            |
    h^2 * Sn = |      -1    2   -1       |
               |           -1    2   -1  |
               |                -2    2  |
    """

    diag_main = np.full(shape=n+1, fill_value=2)

    diag_lower = np.full(shape=n, fill_value=-1)
    diag_lower[-1] = -2

    diag_upper = np.full(shape=n, fill_value=-1)
    diag_upper[0] = -2

    A = sp.diags(
        diagonals=[diag_lower, diag_main, diag_upper],
        offsets=[-1, 0, 1], shape=(n+1, n+1), format='csr')
    
    return A / (h**2)


def laplacian_2d(N: int, M: int, h: float) -> sp.csr_matrix:
    """
    Implements formula on slides #7 page 22
    """
    S_N = Sn_NN(n=N, h=h)
    S_M = Sn_DN(n=M, h=h)
    I_M = sp.eye(M+1, format="csr")
    I_N = sp.eye(N+1, format="csr")

    return (sp.kron(I_M, S_N, format="csr") + sp.kron(S_M, I_N, format="csr"))


def f_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Forcing function f in: -Lapl(u) = f

    Parameters:
        x, y: 2D array of coordinates
    
    Returns:
        2D array of values, shape as input coordinates
    """
    return 100 * np.exp(-0.5 * (x - 4)**2 - 4 * (y - 1)**2)


def get_rhs(X: np.ndarray, Y: np.ndarray, T_ext: float, h: float) -> np.ndarray:
    """
    Returns the RHS vector in the discretised linear system
    """

    rhs = f_grid(X, Y).ravel(order='C')   # reshape forcing function as 1D vector

    # Impose the Dirichlet BC at the bottom side of the domain
    nodes_x_direction = X.shape[1]
    rhs[:(nodes_x_direction)] = T_ext / h**2

    return rhs


# Define the problem parameters
Lx = 12.0
Ly = 5.0
N_VALS = [60, 120, 240, 480, 960]
T_EXT = 25.0
F = 2.0
X_PROBE, Y_PROBE = 6.0, 2.0

probe_vals = []
h_vals = []
for N in N_VALS:
    # Solve PDE
    x, y, X, Y, h, M = create_grid(Lx, Ly, N)
    h_vals.append(h)
    A = laplacian_2d(N=N, M=M, h=h)
    rhs = get_rhs(X=X, Y=Y, T_ext=T_EXT, h=h)
    T = spla.spsolve(A=A, b=rhs)
    T_grid = T.reshape((M+1,N+1), order='C')

    # Compute temperature at probe location
    interp = RegularGridInterpolator((y,x), T_grid)
    probe_val = interp((Y_PROBE, X_PROBE))
    print(f'T(x=6, y=2) for N={N}: ', probe_val)
    probe_vals.append(probe_val)

    # Plot temperature distribution
    if N == N_VALS[1]:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(X=X, Y=Y, Z=T_grid, cmap=cm.viridis, linewidth=0, antialiased=False)
        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('$y$', fontsize=14)
        ax.view_init(40, -30)
        fig.suptitle('Temperature distribution - $f(x,y)=100 \,\\text{exp}\left[-\\frac{1}{2}(x-4)^2-4(y-1)^2\\right]$', fontsize=16)
        plt.tight_layout()
        plt.show()

# Plot convergence
err_est = np.abs(np.diff(np.array(probe_vals)))
plt.figure(figsize=(8, 6))
plt.loglog(h_vals[1:], err_est, marker='o', label='Estimated error')
plt.loglog(h_vals[1:], [0.5*h**2 for h in h_vals[1:]], linestyle='--', label='$\mathcal{O}(h^2)$')
plt.grid()
plt.xlabel('h', fontsize=14)
plt.ylabel('Error estimate', fontsize=14)
plt.title('Convergence plot at probe location', fontsize=16)
plt.legend(fontsize=12)
plt.show()