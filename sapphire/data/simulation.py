"""Simulation data module"""
from collections import deque
from dataclasses import dataclass
from typing import Union, Tuple, Callable, Dict, Any, Deque
from firedrake.bcs import DirichletBC
from sapphire.data.mesh import Mesh
from sapphire.data.solution import Solution
from sapphire.data.problem import Problem
from sapphire.data.solver import Solver
from firedrake import FiniteElement, VectorElement, MixedElement, MixedVectorSpaceBasis


@dataclass
class Simulation:
    """Simulation data class"""

    solutions: Deque[Solution]
    """Solution data

    A solution is saved for each point in the time discretization stencil.
    The deque of solutions is arranged with the latest first and the earliest last.
    The latest solution's time is the time that will be solved first.
    The latest solution's initial values will be used as the initial guess for the nonlinear solver.
    """

    problem: Problem
    """Data for setting up the nonlinear problem"""

    solver: Solver
    """Data for setting up the nonlinear solver"""

    def __init__(
            self,
            mesh: Mesh,
            element: Union[FiniteElement, VectorElement, MixedElement],
            time_discretization_stencil_size: int,
            solution_component_names: Tuple[str],
            residual: Callable[[Tuple[Solution]], Any],
            dirichlet_boundary_conditions: Callable[[Solution], Tuple[DirichletBC]],
            ufl_constants: Dict[str, float],
            firedrake_solver_parameters: dict,
            nullspace: Union[Callable[[Solution], MixedVectorSpaceBasis], None] = None,
            quadrature_degree: Union[int, None] = None):

        if time_discretization_stencil_size < 1:

            raise Exception("'time_discretization_stencil_size' must be at least 1 (which would be for steady state simulation).")

        solutions = []

        for i in range(time_discretization_stencil_size):

            if 'timestep_size' in ufl_constants:

                time = -i*ufl_constants['timestep_size']

            else:

                time = None

            solutions.append(Solution(
                mesh=mesh,
                element=element,
                component_names=solution_component_names,
                ufl_constants=ufl_constants,
                quadrature_degree=quadrature_degree,
                time=time,
                checkpoint_index=-i))

        self.solutions = deque(solutions)

        self.problem = Problem(
            residual=residual,
            dirichlet_boundary_conditions=dirichlet_boundary_conditions)

        if nullspace is None:

            self.solver = Solver(firedrake_solver_parameters=firedrake_solver_parameters)

        else:

            self.solver = Solver(firedrake_solver_parameters=firedrake_solver_parameters, nullspace=nullspace(self.solutions[0]))
