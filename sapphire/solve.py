"""Module for solving nonlinear problems"""
from sapphire.data.simulation import Simulation
from firedrake import NonlinearVariationalProblem, NonlinearVariationalSolver, derivative, ConvergenceError


def solve(sim: Simulation):
    """Set up the problem and solver, and solve.

    This ensures that the problem and solver setup are up-to-date before calling the solver.
    All compiled objects are cached, so this does not have any significant performance overhead.

    `sim.solutions[0]` will be modified with the result.
    """
    r = sim.problem.residual(tuple(sim.solutions))

    solution = sim.solutions[0]

    solver = NonlinearVariationalSolver(
        problem=NonlinearVariationalProblem(
            F=r,
            u=solution.function,
            bcs=sim.problem.dirichlet_boundary_conditions(solution),
            J=derivative(r, solution.function)),
        nullspace=sim.solver.nullspace,
        solver_parameters=sim.solver.firedrake_solver_parameters)

    print("Solving nonlinear problem with {} degrees of freedom".format(solution.function.vector().size()))

    try:

        solver.solve()

        solution.snes_cumulative_iteration_count += snes_iteration_count(solver)

        solution.solved = True

    except (ConvergenceError) as exception:

        solution.snes_cumulative_iteration_count += snes_iteration_count(solver)

        solution.solved = False

        raise exception


def snes_iteration_count(solver: NonlinearVariationalSolver) -> int:
    """ Get iteration number without losing type info """
    return solver.snes.getIterationNumber()
