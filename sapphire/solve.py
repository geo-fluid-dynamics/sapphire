"""Module for solving nonlinear problems"""
from sapphire.data.solution import Solution
from sapphire.data.simulation import Simulation
from firedrake import NonlinearVariationalProblem, NonlinearVariationalSolver, derivative


def solve(sim: Simulation) -> Solution:
    """Set up the problem and solver, and solve.

    This ensures that the problem and solver setup are up-to-date before calling the solver.
    All compiled objects are cached, so this does not have any significant performance overhead.
    """
    r = sim.problem.residual(tuple(sim.solutions))

    solution = sim.solutions[0]

    solver = NonlinearVariationalSolver(
        problem=NonlinearVariationalProblem(
            F=r,
            u=solution.function,
            bcs=sim.problem.dirichlet_boundary_conditions,
            J=derivative(r, solution.function)),
        nullspace=sim.solver.nullspace,
        solver_parameters=sim.solver.firedrake_solver_parameters)

    solver.solve()

    solution.snes_cumulative_iteration_count += snes_iteration_count(solver)

    return solution


def snes_iteration_count(solver: NonlinearVariationalSolver) -> int:
    """ Get iteration number without losing type info """
    return solver.snes.getIterationNumber()
