"""Solver data module"""
from dataclasses import dataclass
from typing import Union
from firedrake import MixedVectorSpaceBasis


DEFAULT_FIREDRAKE_SOLVER_PARAMTERS = {
    'snes_type': 'newtonls',
    'snes_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'mat_type': 'aij',
    'pc_factor_mat_solver_type': 'mumps'}


@dataclass
class Solver:
    """Solver data class

    This is primarily used for setting up Firedrake's `NonlinearVariationalSolver`, but it could be extended with sapphire specific solution procedure options, e.g. for continuation.
    """
    nullspace: Union[MixedVectorSpaceBasis, None] = None

    firedrake_solver_parameters: dict = None
    """This will be used as Firedrake's `solver_parameters` e.g. for PETSc configuration"""

    def __post_init__(self):

        if self.firedrake_solver_parameters is None:

            self.firedrake_solver_parameters = DEFAULT_FIREDRAKE_SOLVER_PARAMTERS
