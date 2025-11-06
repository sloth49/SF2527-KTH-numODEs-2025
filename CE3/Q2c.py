# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 3, Part 2-c
#
# Author: Alessio / Tim
# Date: 3 November 2025
# -----------------------------------------------------------------------------

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from solver_system import Solver, NumericalSchemes
from domain import Domain

def main():
     # Problem parameters
     L0 = -0.4
     L1 = 0.7
     Fr = 0.35
     ALPHA = 1 / Fr
     LAMBDA1 = 1 + ALPHA
     LAMBDA2 = 1 - ALPHA
     T_FINAL = 0.15




if __name__ == "__main__":
     main() 
