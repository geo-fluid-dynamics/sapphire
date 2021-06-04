""" Solver module """
import sapphire.helpers
import sapphire.data
import firedrake as fe


def nonlinear_solve(problem: sapphire.data.Problem) -> sapphire.data.Solution:
    """Set up the problem and solver, and solve.

    This ensures that the problem and solver setup are up-to-date before calling the solver.
    All compiled objects are cached, so this does not have any significant performance overhead.
    """
    r = problem.residual(problem.solution)

    solver = fe.NonlinearVariationalSolver(
        problem=fe.NonlinearVariationalProblem(
            F=r,
            u=problem.solution.function,
            bcs=problem.dirichlet_boundary_conditions,
            J=fe.derivative(r, problem.solution.function)),
        nullspace=problem.nullspace,
        solver_parameters=problem.solver_parameters)

    solver.solve()

    problem.solution.snes_cumulative_iteration_count += sapphire.helpers.snes_iteration_count(solver)

    return problem.solution
