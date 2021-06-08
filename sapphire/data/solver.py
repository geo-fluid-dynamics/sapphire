"""Solver data module"""
from dataclasses import dataclass
from typing import Union
from firedrake import MixedVectorSpaceBasis


@dataclass
class Solver:
    """Solver data class

    This is primarily used for setting up Firedrake's `NonlinearVariationalSolver`, but it could be extended with sapphire specific solution procedure options, e.g. for continuation.
    """

    firedrake_solver_parameters: dict
    """This will be used as Firedrake's `solver_parameters` e.g. for PETSc configuration"""

    nullspace: Union[MixedVectorSpaceBasis, None] = None
