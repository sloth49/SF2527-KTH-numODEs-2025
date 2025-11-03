# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 3, Part 2-c
#
# Author: Alessio / Tim
# Date: 3 November 2025
# -----------------------------------------------------------------------------

import numpy as np
from solver import Solver, NumericalSchemes
from domain import Domain

if __name__ == "__main__":
    # Problem parameters
    L0 = -0.4
    L1 = 0.7
    Fr = 0.35
    ALPHA = 1 / Fr
    T_FINAL = 0.15

    s = (lambda x:
         np.sin(20 * np.pi * x)
         if np.abs(x) < 1 / 20
         else 0)
    r = (lambda t:
         1
         if np.sin((40 * np.pi * t) + (np.pi / 6)) > 0.5
         else 0)
    f = lambda x, t, s, r: s(x) * r(t)

 
