""" Module for running a Simulation. """
from logging import warning
import typing
import sapphire.helpers
import sapphire.data
import sapphire.nonlinear_solve
import sapphire.output.plot


ENDTIME_TOLERANCE = 1.e-8
""" Allows endtime to be only approximately reached.

This is larger than a typical floating point comparison tolerance because errors accumulate between timesteps.
"""


def default_solve(problem: sapphire.data.Problem) -> sapphire.data.Solution:

    return sapphire.nonlinear_solve.nonlinear_solve(problem)


def default_output(solution: sapphire.data.Solution, outdir_path: str):

    sapphire.output.plot.plot(solution=solution, outdir_path=outdir_path)


def run(  # pylint: disable=too-many-arguments
        sim: sapphire.data.Simulation,
        endtime: float = None,
        solve: typing.Callable[[sapphire.data.Problem], sapphire.data.Solution] = None,
        postprocess: typing.Callable[[sapphire.data.Solution], sapphire.data.Solution] = None,
        output: typing.Callable[[sapphire.data.Solution], None] = None,
        validate: typing.Callable[[sapphire.data.Solution], bool] = None,
        ) -> sapphire.data.Simulation:
    """Run simulation forward in time.

    :param sim: The simulation data.

    :param endtime: Run the simulation until reaching this time. Use default value None for steady state problems.

    :param solve: This is called to solve each time step.

    :param postprocess: This is called to post-process the simulation data before writing outputs.

    :param output: This is called to write outputs after solving each time step.

    :param validate: This is called to validate data before solving.
    """
    for function, default_function, name in zip((solve, output), (default_solve, default_output), ('solve', 'output')):

        if function is None:

            function = default_function

            warning("`run` is using default `{}` function".format(name))

    if validate:

        print("Validating all initial solution data")

        for solution in sim.solutions:

            valid = validate(solution)

            if not valid:

                raise Exception("Solution data failed validation")

    time = sim.solutions[0].time

    if time is None:

        if endtime is not None:

            raise Exception("`endtime` was specified but `solution.time` is None")

        if len(sim.solutions) > 1:

            raise Exception("`sim.solutions` should only contain a single solution if there is no time discretization")

        sim.solutions[0] = _run_one_step(problem=sim.problem, solve=solve, postprocess=postprocess, output=output, validate=validate)

        return sim

    #
    starttime = sim.solutions[1].time

    first_time_to_solve = sim.solutions[0].time

    timestep_size = first_time_to_solve - starttime

    stepcount = 0

    time = starttime

    while time <= (endtime + ENDTIME_TOLERANCE):

        if stepcount > 0:

            sim.solutions = sapphire.helpers.rotate_deque(sim.solutions, 1)

            sim.solutions[0].function = sapphire.helpers.assign_function_values(sim.solutions[1].function, sim.solutions[0].function)

            sim.solutions[0].time = sim.solutions[1].time + timestep_size

            sim.solutions[0].checkpoint_index = sim.solutions[1].checkpoint_index + 1

        sim.solutions[0] = _run_one_step(problem=sim.problem, solve=solve, postprocess=postprocess, output=output, validate=validate)

        stepcount += 1

        time = starttime + stepcount*timestep_size

    return sim


def _run_one_step(
        problem: sapphire.data.Problem,
        solve: typing.Callable[[sapphire.data.Problem], sapphire.data.Solution] = None,
        postprocess: typing.Callable[[sapphire.data.Solution], sapphire.data.Solution] = None,
        output: typing.Callable[[sapphire.data.Solution], None] = None,
        validate: typing.Callable[[sapphire.data.Solution], bool] = None,
        ) -> sapphire.data.Solution:

    solution = solve(problem)

    if solution.time:

        print("Solved at time t = {}".format(solution.time))

    if postprocess:

        print("Postprocessing solution data")

        solution = postprocess(solution)

    print("Writing outputs")

    output(solution)

    if validate:

        print("Validating latest solution data")

        valid = validate(solution)

        if not valid:

            raise Exception("Solution data failed validation")

    return solution
