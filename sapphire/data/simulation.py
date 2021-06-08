"""Simulation data module"""
from collections import deque
from dataclasses import dataclass
from typing import Union, Tuple, Callable, Dict, Any, Deque
from firedrake.bcs import DirichletBC
from sapphire.data.mesh import Mesh
from sapphire.data.solution import Solution
from sapphire.data.problem import Problem
from sapphire.data.solver import Solver
from firedrake import Constant, Function, FiniteElement, VectorElement, MixedElement, MixedVectorSpaceBasis


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
            solution_component_names: Tuple[str],
            residual: Callable[[Tuple[Solution]], Any],
            dirichlet_boundary_conditions: Callable[[Solution], Tuple[DirichletBC]],
            ufl_constants: Dict[str, Constant],
            firedrake_solver_parameters: dict,
            initial_times: Union[Tuple[float], None],
            initial_values_functions: Union[Tuple[Function], None] = None,
            nullspace: Union[Callable[[Solution], MixedVectorSpaceBasis], None] = None,
            quadrature_degree: Union[int, None] = None):

        solutions = []

        if initial_times is None:

            _initial_times = (None,)

        else:

            _initial_times = initial_times

        if initial_values_functions is None:

            _initial_values_functions = (None,)*len(_initial_times)

        else:

            _initial_values_functions = initial_values_functions

        for i, time in enumerate(_initial_times):

            solution = Solution(
                mesh=mesh,
                element=element,
                component_names=solution_component_names,
                ufl_constants=ufl_constants,
                quadrature_degree=quadrature_degree,
                time=time)

            iv = _initial_values_functions[i]

            if iv is not None:

                solution.function.assign(iv)

            solutions.append(solution)

        self.solutions = deque(solutions)

        self.problem = Problem(
            residual=residual,
            dirichlet_boundary_conditions=dirichlet_boundary_conditions(self.solutions[0]))

        if nullspace is None:

            self.solver = Solver(firerake_solver_parameters=firedrake_solver_parameters)

        else:

            self.solver = Solver(firedrake_solver_parameters=firedrake_solver_parameters, nullspace=nullspace(solution))
