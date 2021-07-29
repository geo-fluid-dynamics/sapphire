"""Nonlinear solver continuation

for regularized nonlinear problems.
"""
from typing import Callable, Tuple, Union
from sapphire.data.simulation import Simulation, Solution
from firedrake import Function, Constant, ConvergenceError


class ContinuationError(Exception):

    pass


def find_working_continuation_parameter_value(
        sim: Simulation,
        solve: Callable[[Simulation], None],
        continuation_parameter_and_name: Tuple[Constant, str],
        search_operator: Callable = lambda r: 2.*r,
        max_attempts: int = 8,
        backup_solution_function: Union[Function, None] = None,
        output: Union[Callable[[Solution], None], None] = None,
        ) -> float:
    """ Attempt to solve a sequence of nonlinear problems where the continuation parameter value is varied according to the search operator until a solution is found.

    The resulting solution will *not* be the solution to the original problem with the original parameter value.
    Rather the solution can be used as a starting point for bounded continuation.
    """
    solution = sim.solutions[0]

    continuation_parameter, rname = continuation_parameter_and_name

    if continuation_parameter not in solution.ufl_constants:

        raise Exception("The continuation parameter must be one of the solution's UFL constants.")

    if backup_solution_function is None:

        backup_solution_function = Function(solution.function_space)

        backup_solution_function.assign(solution.function)

    else:

        if backup_solution_function.function_space() != solution.function_space:

            raise Exception("The backup solution function must use the same function space as the solution function.")

        backup_solution_function.assign(solution.function)

    r0 = continuation_parameter.__float__()

    r = r0

    print("Searching for a working continuation parameter value")

    for attempt in range(max_attempts):

        continuation_parameter.assign(r)

        print("Trying {} = {}".format(rname, r))

        try:

            snes_iteration_count = solution.snes_cumulative_iteration_count

            solve(sim)

            print("Solved with {} = {}".format(rname, r))

            solution.continuation_history.append((rname, r, solution.snes_cumulative_iteration_count - snes_iteration_count))

            if output:

                print("Writing output for this intermediate solution")

                output(solution)

            return r

        except (ConvergenceError, ContinuationError) as exception:

            r = search_operator(r)

            solution.function.assign(backup_solution_function)

            if attempt == range(max_attempts)[-1]:

                continuation_parameter.assign(r0)

                raise ContinuationError("Failed to find working continuation parameter value after applying search operator {} times".format(max_attempts)) from exception

    raise ContinuationError("Failed to find working value for continuation parameter before exceeding maximum number of attempts ({})".format(max_attempts))


def solve_with_bounded_continuation_sequence(  # pylint: disable=too-many-arguments
        sim: Simulation,
        solve: Callable[[Simulation], None],
        continuation_parameter_and_name: Tuple[Constant, str],
        initial_sequence: Tuple[float],
        maxcount: int = 16,
        start_index: int = 0,
        backup_solution_function: Union[Function, None] = None,
        output: Union[Callable[[Solution], None], None] = None,
        ):
    """ Solve a sequence of nonlinear problems where the continuation parameter value varies between bounds.

    Always continue from left to right.

    If successful, then the final solution is for the right bounding continuation parameter value.

    `sim.solutions[0]` will be modified with the result.
    """
    solution = sim.solutions[0]

    continuation_parameter, rname = continuation_parameter_and_name

    if continuation_parameter not in solution.ufl_constants:

        raise Exception("The continuation parameter must be one of the solution's UFL constants")

    r0 = continuation_parameter.__float__()

    if initial_sequence[-1] != r0:

        raise Exception("The sequence must end with the actual parameter value.")

    sequence = initial_sequence

    first_r_to_solve = sequence[start_index]

    attempts = range(maxcount - len(sequence))

    solved = False

    if backup_solution_function is None:

        backup_solution_function = Function(solution.function_space)

        backup_solution_function.assign(solution.function)

    else:

        if backup_solution_function.function_space() != solution.function_space:

            raise Exception("The backup solution function must use the same function space as the solution function.")

        backup_solution_function.assign(solution.function)

    for attempt in attempts:

        r_start_index = sequence.index(first_r_to_solve)

        try:

            for r in sequence[r_start_index:]:

                continuation_parameter.assign(r)

                print("Trying to solve with continuation parameter {} = {}".format(rname, r))

                snes_iteration_count = solution.snes_cumulative_iteration_count

                solve(sim)

                solution.continuation_history.append((rname, r, solution.snes_cumulative_iteration_count - snes_iteration_count))

                backup_solution_function.assign(solution.function)

                print("Solved with continuation parameter {} = {}".format(rname, r))

                if output:

                    print("Writing output for this intermediate solution")

                    output(solution)

            solved = True

            break

        except (ConvergenceError, ContinuationError) as exception:

            current_r = continuation_parameter.__float__()

            rs = sequence

            print("Failed to solve with continuation parameter {} = {} from the sequence {}".format(rname, current_r, rs))

            index = rs.index(current_r)

            if attempt == attempts[-1]:

                continuation_parameter.assign(r0)

                raise ContinuationError("Failed to solve with maximum allowed continuation points {}".format(maxcount)) from exception

            if index == 0:

                continuation_parameter.assign(r0)

                raise ContinuationError("Failed to solve for first continuation parameter value, {} = {}, from the provided sequence".format(rname, current_r)) from exception

            solution.function.assign(backup_solution_function)

            r_to_insert = (current_r + rs[index - 1])/2.

            new_rs = rs[:index] + (r_to_insert,) + rs[index:]

            sequence = new_rs

            print("Inserted new value of {} = {}".format(rname, r_to_insert))

            first_r_to_solve = r_to_insert

    if not solved:

        raise ContinuationError("Failed to solve with continuation")

    if not continuation_parameter.__float__() == r0:

        raise Exception("Invalid state")

    if not sequence[-1] == r0:

        raise Exception("Invalid state")

    solution.extras[rname+'_continuation_sequence'] = tuple(sequence)


def solve_with_timestep_size_continuation(sim: Simulation, solve: Callable[[Simulation], None], maxcount):

    ufl_timestep_size = sim.solutions[0].ufl_constants.timestep_size

    solve_with_bounded_continuation_sequence(  # pylint: disable=too-many-arguments
        sim=sim,
        solve=solve,
        continuation_parameter_and_name=(ufl_timestep_size, 'Delta_t'),
        initial_sequence=(0, ufl_timestep_size.__float__()),
        maxcount=maxcount,
        start_index=-1,  # Try the actual timestep size first
        )
