# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 3, Part 2-c
#
# Author: Alessio / Tim
# Date: 6 November 2025
# -----------------------------------------------------------------------------
import numpy as np
from domain import Domain
from typing import Callable
from enum import Enum
import sys
from mpl_toolkits.mplot3d import Axes3D

class NumericalSchemes(Enum):
    UPWIND = "Upwind"
    LAX_WENDROFF = "Lax-Wendroff"


class SolverSystem:
    """
    A finite difference solver to solve conservation equations of the type:
                
        where 

    Parameters:
    
    Attributes:
    """
    DIM = 2
    initial_cond: np.ndarray | None # shape: (DIM, Nx+1)
    sol: np.ndarray | None # shape: (DIM, Nx+1, Nt+1)
    _A: np.ndarray # shape: (DIM, DIM)
    _Aplus: np.ndarray # shape: (DIM, DIM)
    _Lambda_plus: np.ndarray # shape: (DIM, DIM)
    _Lambda_minus: np.ndarray # shape: (DIM, DIM)
    _char_speeds: np.ndarray # shape: (DIM,)
    _Co: float | None
    _domain: Domain | None
    _F: list[Callable[[np.ndarray, np.ndarray], float] | float]
    _S: np.ndarray # shape: (DIM, DIM)
    _solver_called: bool | None

    def __init__(
            self, A: np.ndarray,
            F: list[Callable[[float, float], float] | float]
        ) -> None:
        if not A.shape == (self.DIM, self.DIM):
            raise ValueError(f"A matrix must be a {self.DIM}x{self.DIM} matrix")
        self._A = A
        if not len(F) == self.DIM:
            raise ValueError(f"F must be a {self.DIM}-elements list")
        self._F = F
        self._char_speeds, self._S = np.linalg.eig(A)
        Lambda = np.diag(self._char_speeds)
        self._Lambda_plus = np.where((Lambda > 0), Lambda, 0)
        self._Lambda_minus = np.where((Lambda < 0), Lambda, 0)
        self.initial_cond = None
        self._Co = None
        self._domain = None
        self._solver_called = None
        self.sol = None


    def cfl_condition_satisfied(self) -> bool:
        if self._domain is None:
            raise ValueError("Domain not set. Cannot check CFL condition.")
        else:
            max_char_speed = np.max(np.abs(self._char_speeds))
            self._Co = (max_char_speed * self._domain.dt) / self._domain.dx
            return (self._Co <= 1) # type: ignore
    
    def get_Fgrid(self):
        """
        Returns vector-valued forcing function F(x,y) on a X,Y mesh grid.
        Shape: (DIM, len(x), len(y))
        """
        X, Y = np.meshgrid(self._domain.x, self._domain.t, indexing='ij') # type: ignore
        return np.stack([f(X, Y) for f in self._F]) # type: ignore


    def solve_pde(
            self,
            domain: Domain,
            initial_condition: np.ndarray,
            num_scheme: NumericalSchemes
            ) -> np.ndarray:
        """
        Solve the PDE

        Parameters:
        Returns:
        """
        self._domain = domain

        # check stability, exit if not ok.
        if not self.cfl_condition_satisfied():
            print(f"CFL condition not satisfied (Co={self._Co}). "
                  + "Stopping the program.")
            sys.exit()
        print("Solving the PDE with the " + num_scheme.value
              + f" scheme, Courant No: {self._Co}.")
        self._solver_called = True

        mu = domain.dt / domain.dx
        dt = self._domain.dt    # local alias for readability
        F_grid = self.get_Fgrid()   # forcing function over x-t domain
        self.initial_cond = initial_condition
        # initialise solution container
        self.sol = np.zeros(
            shape=(
                initial_condition.shape[0],
                len(domain.x),
                len(domain.t)
            )
        )
        # Create groups of node indices for scheme stencil.
        # To be used on time-level arrays of shape (DIM, len(x))
        interior = slice(1, -1)
        east = slice(2, None)
        west = slice(None, -2)
        
        # === Switch the solver to the specified numerical scheme ===
        if num_scheme == NumericalSchemes.UPWIND:
            S_inv = np.linalg.inv(self._S)
            A_plus = self._S @ self._Lambda_plus @ S_inv
            A_minus = self._S @ self._Lambda_minus @ S_inv
            def step_UW(u: np.ndarray, F_old: np.ndarray) -> np.ndarray:
                u_new = np.copy(u)
                u_new[:, interior] = (
                    u[:, interior]
                    - mu * A_plus @ (u[:, interior] - u[:, west])
                    - mu * A_minus @ (u[:, east] - u[:, interior])
                    + dt * F_old[:, interior]
                )
                return u_new
        
        elif num_scheme == NumericalSchemes.LAX_WENDROFF:
            A = self._A     # local alias for readability
            def step_LW(
                    u: np.ndarray, F: np.ndarray, F_new: np.ndarray
                ) -> np.ndarray:
                F_tilde = (
                    0.5 * (F[:, interior] + F_new[:, interior])
                    - 0.25 * mu * A @ (F[:, east] - F[:, west])
                )
                u_new = np.copy(u)
                u_new[:, interior] = (
                    u[:, interior]
                    - 0.5 * mu * A @ (u[:, east] - u[:, west]) # type: ignore
                    + 0.5 * mu**2 * A @ A @ (u[:, east] - 2 * u[:, interior] + u[:, west]) # type: ignore
                    + dt * F_tilde
                )
                return u_new
        
        else:
                raise ValueError(f"Unkonwn numerical scheme: {num_scheme}")

        # Initial condition
        self.sol[:, :, 0] = self.initial_cond
        u_old = self.initial_cond
            
        # Time marching loop
        for t_step_curr in range(1, domain.Nt+1):            
            t_curr = domain.t[t_step_curr]      # n+1 time level
            F_old = F_grid[:, :, t_step_curr-1]  # F^n
            if num_scheme == NumericalSchemes.UPWIND:
                u_curr = step_UW(u=u_old, F_old=F_old) # type: ignore
            elif num_scheme == NumericalSchemes.LAX_WENDROFF:
                F_curr = F_grid[:, :, t_step_curr]  # F^{n+1}
                u_curr = step_LW(u=u_old, F=F_old, F_new=F_curr) # type: ignore
            u_curr[:, 0] = 2 * u_curr[:, 1] - u_curr[:, 2]    # Numerical BC at LHS boundary
            u_curr[:, -1] = 2 * u_curr[:, -2] - u_curr[:, -3]    # Numerical BC at RHS boundary
            self.sol[:, :, t_step_curr] = u_curr     # store the result
            u_old = u_curr
        
        return self.sol