# -----------------------------------------------------------------------------
# SF2527 Numerical Methods for Differential Equations I
# Computer Exercise 3, Part 1-c
#
# Author: Alessio / Tim
# Date: 20 October 2025
# -----------------------------------------------------------------------------
import numpy as np
from domain import Domain
from typing import Callable
from enum import Enum
import sys
from mpl_toolkits.mplot3d import Axes3D

class NumericalSchemes(Enum):
    UPWIND = "Upwind"
    LAX_FREDRICHS = "Lax-Friedrichs"
    LAX_WENDROFF = "Lax-Wendroff"


class Solver:
    """
    A finite difference solver to solve conservation equations of the type:
                (u)_t + a * (u)_x = 0
        where u(x,t) is a transported conservative quantity.

    Parameters:
        a (float):
            advection speed (> 0)
        domain (Domain):
            space-time domain
        initial_cond (ndarray):
            initial condition, shape `(len(_domain.x),)`
    
    Attributes:
        _Co (float):
            (a * dt / dx), Courant number
        _left_bc (`Callable[[float], float]`):
            Function of time returning the left boundary condition, u(0,t)
        u (ndarray):
            solution array, shape `(domain.t, domain.x)
    """
    initial_cond: np.ndarray | None
    u: np.ndarray | None
    _a: float
    _Co: float | None
    _domain: Domain | None
    _left_bc: Callable[[float], float] | None
    _solver_called: bool | None

    def __init__(self, a :float = 1) -> None:
        if a <= 0:
            raise ValueError("Advection speed 'a' must be positive.")
        self._a = a
        self.initial_cond = None
        self._Co = None
        self._domain = None
        self._left_bc = None
        self._solver_called = None
        self.u = None


    def cfl_condition_satisfied(self) -> bool:
        if self._domain is None:
            raise ValueError("Domain not set. Cannot check CFL condition.")
        
        self._Co = (self._a * self._domain.dt) / self._domain.dx
        return (self._Co <= 1)
    

    def solve_pde(
            self,
            domain: Domain,
            initial_condition: np.ndarray,
            left_bc: Callable[[float], float],
            num_scheme: NumericalSchemes
            ) -> np.ndarray:
        """
        Solve the PDE in U(x,t):
            Ut + aUx = 0  (a = const > 0)
        using a specified numerical scheme.

        Parameters:
            domain (Domain):
                the discretised space-time domain
            initial_condition (np.ndarray):
                U(x,0), shape`(domain.x.shape)`
            left_bc(Callabe[[float], float]):
                function of t for the left BC
            num_scheme (NumericalScheme):
                the numerical scheme to use for solving
        Returns:
            u (np.ndarray):
                2D array,
                shape`(domain.t.shape, domain.x.shape)`
        """
        self._domain = domain
        self._left_bc = left_bc
        self.u = np.zeros(shape=domain.shape)  # initialise solution container
        self.initial_cond = initial_condition

        # check stability, exit if not ok.
        if not self.cfl_condition_satisfied():
            print(f"CFL condition not satisfied (Co={self._Co}). Stopping the program.")
            sys.exit()
        Co = self._Co   # Local alias for readability
        print("Solving the PDE with the "
              + num_scheme.value
              + f" scheme, Courant No: {Co}.")
        self._solver_called = True
        
        # === Switch the solver to the specified numerical scheme ===
        if num_scheme == NumericalSchemes.UPWIND:
            # create groups of node indices for scheme stencil
            east = slice(1, None)
            west = slice(None, -1)
            def step(u: np.ndarray) -> np.ndarray:
                u_new = np.copy(u)
                u_new[east] = u[east] - Co * (u[east] - u[west])
                return u_new
            
        elif num_scheme == NumericalSchemes.LAX_FREDRICHS:
            # Create groups of node indices for scheme stencil
            interior = slice(1, -1)
            east = slice(2, None)
            west = slice(None, -2)
            def step(u: np.ndarray) -> np.ndarray:
                u_new = np.copy(u)
                u_new[interior] = (
                    0.5 * (u[east] + u[west])
                    -0.5 * Co * (u[east] - u[west]) # type: ignore
                )
                return u_new
        
        elif num_scheme == NumericalSchemes.LAX_WENDROFF:
            # Create groups of node indices for scheme stencil
            interior = slice(1, -1)
            east = slice(2, None)
            west = slice(None, -2)
            def step(u: np.ndarray) -> np.ndarray:
                u_new = np.copy(u)
                u_new[interior] = (
                    u[interior]
                    - 0.5 * Co * (u[east] - u[west]) # type: ignore
                    + 0.5 * Co**2 * (u[east] - 2 * u[interior] + u[west]) # type: ignore
                )
                return u_new
        
        else:
                raise ValueError(f"Unkonwn numerical scheme: {num_scheme}")

        # Initial condition
        self.u[0, :] = self.initial_cond
        u_old = self.initial_cond
            
        # Time marching loop
        for t_step_curr in range(1, domain.Nt+1):            
            t_curr = domain.t[t_step_curr]      # time level to calculate the solution at
            u_curr = step(u_old)                # march the solution forward
            u_curr[0] = self._left_bc(t_curr)   # apply BC at left boundary
            if num_scheme != NumericalSchemes.UPWIND:
                u_curr[-1] = 2 * u_curr[-2] - u_curr[-3]    # Numerical BC at RHS boundary
            self.u[t_step_curr, :] = u_curr     # store the result
            u_old = u_curr
        
        return self.u