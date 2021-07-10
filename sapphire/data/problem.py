"""Problem data module"""
from dataclasses import dataclass
from typing import Callable, Tuple, Any
from sapphire.data.solution import Solution
from firedrake import DirichletBC


DEFAULT_SOLVER_PARAMTERS = {
    'snes_type': 'newtonls',
    'snes_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'mat_type': 'aij',
    'pc_factor_mat_solver_type': 'mumps'}


@dataclass
class Problem:
    """Problem data class"""
    residual: Callable[[Tuple[Solution]], Any]  # @todo What is the type returned by `* fe.dx`?
    """The residual corresponding to the weak form governing equations.

    This is Callable because the residual must be updated whenever the solution deque rotates to avoid excessive copying of solution function values. """

    dirichlet_boundary_conditions: Callable[[Solution], Tuple[DirichletBC]]
