# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 3, Part 1-c
#
# Author: Alessio / Tim
# Date: 20 October 2025
# -----------------------------------------------------------------------------
import numpy as np
import sys
from dataclasses import dataclass

@dataclass
class Domain:
    """
    A data class to define the space-time domain for the solver.

    Parameters:
        L:      length of domain [m]
        T:      final time [s]
        Nx:     number of grid cells
        Nt:     number of time steps
    """
    L: float
    T: float
    Nx: int
    Nt: int

    @property
    def dx(self) -> float:
        return self.L / self.Nx
    
    @property
    def dt(self) -> float:
        return self.T / self.Nt
    
    @property
    def x(self) -> np.ndarray:
        return np.linspace(start=0, stop=self.L, num=self.Nx+1)
    
    @property
    def t(self) -> np.ndarray:
        return np.linspace(start=0, stop=self.T, num=self.Nt+1)
    
    @property
    def shape(self) -> tuple[int, int]:
        return (self.Nt+1, self.Nx+1)
    

    def get_meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.t, indexing='xy')
    
    
    def __post_init__(self):
        if (self.Nx <= 0) or (self.Nt <= 0):
            raise ValueError("Nx and Nt must be positive integers.")
        if (self.L <= 0) or (self.T <= 0):
            raise ValueError("L and T must be positive values.")