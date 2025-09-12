# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 1, Part 4a
#
# Author: Alessio / Tim
# Date: 26 August 2025
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve


def create_grid(Lx: float, Ly: float, N: int):             
    x = np.linspace(start=0, stop=Lx, num=N+1)
    h = x[1] - x[0]
    M = int(Ly / h)
    y = np.linspace(start=0, stop=Ly, num=M+1)
    X, Y = np.meshgrid(x, y)
    return X, Y, h, M


def Sn(n: int, h: float) -> sp.csr_matrix:
    """
    One-dimensional finite-difference derivative matrix 
    of size (n-1) x (n-1) for second derivative:

    For n=5, for example:
 
    h^2 * Sn =
         2    -1      0     0     0
        -1     2     -1     0     0
         0    -1      2    -1     0
         0     0     -1     2    -1
         0     0      0    -1     2

    Lecture notes reference: handout #7 page 22
    """

    diag_main = np.full(n-1, 2)
    diag_off = np.full(n-2, -1)
    A = sp.diags([diag_main, diag_off, diag_off],
                 [0, 1, -1], shape=(n-1, n-1), format='csr')
    
    return A / (h**2)


def laplacian_2d(N: int, M: int, h: float) -> sp.csr_matrix:
    S_N = Sn(n=N, h=h)
    S_M = Sn(n=M, h=h)    
    Lp = (
        sp.kron(sp.eye(M-1, format="csr"), S_N, format="csr")
        + sp.kron(S_M, sp.eye(N-1, format="csr"), format="csr")
    )
    return Lp


# Define the problem parameters
Lx = 12.0
Ly = 5.0
N = 60
T_EXT = 25.0
F = 2.0
X_PROBE, Y_PROBE = 6.0, 2.0

# Grid generation
X, Y, h, M = create_grid(Lx, Ly, N)
A = laplacian_2d(N=N, M=M, h=h)

# Forcing function