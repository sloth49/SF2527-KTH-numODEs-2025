# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 1, Part 4a
#
# Author: Alessio / Tim
# Date: 26 August 2025
# -----------------------------------------------------------------------------

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve
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

    return X, Y, h, M


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
        offsets=[-1, 0, 1], shape=(n+1, n+1), format='lil')
    
    return (A.tocsr()) / (h**2)


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
        offsets=[-1, 0, 1], shape=(n+1, n+1), format='lil')
    
    return (A.tocsr()) / (h**2)


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
    return np.full_like(a=x, fill_value=2)


def get_rhs(X: np.ndarray, Y: np.ndarray, T_ext: float, h: float) -> np.ndarray:
    """
    docstring
    """

    rhs = f_grid(X, Y).ravel(order='C')

    n = X.shape[1] - 1
    rhs[:(n+1)] = T_ext / h**2

    return rhs    


# Define the problem parameters
Lx = 12.0
Ly = 5.0
N = 60
T_EXT = 25.0
F = 2.0
X_PROBE, Y_PROBE = 6.0, 2.0

# Solve PDE
X, Y, h, M = create_grid(Lx, Ly, N)
A = laplacian_2d(N=N, M=M, h=h)
rhs = get_rhs(X=X, Y=Y, T_ext=T_EXT, h=h)
T = spla.spsolve(A=A, b=rhs)
T_grid = T.reshape((M+1,N+1), order='C')

# Plot temperature distribution
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(X=X, Y=Y, Z=T_grid, cmap=cm.viridis, linewidth=0, antialiased=False)
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$y$', fontsize=14)
ax.view_init(40, -30)
fig.suptitle('Temperature distribution - uniform heating', fontsize=16)
plt.tight_layout()
plt.show()