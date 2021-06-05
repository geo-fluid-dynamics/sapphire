"""Module for running simulations"""
from logging import warning
from typing import Callable, Deque
from sapphire.data.solution import Solution
from sapphire.data.simulation import Simulation
from sapphire.solve import solve as default_solve
from sapphire.output.plot import plot


ENDTIME_TOLERANCE = 1.e-8
""" Allows endtime to be only approximately reached.

This is larger than a typical floating point comparison tolerance because errors accumulate between timesteps.
"""


def default_output(solution: Solution, outdir_path: str) -> None:

    plot(solution=solution, outdir_path=outdir_path)


def run(  # pylint: disable=too-many-arguments
        sim: Simulation,
        endtime: float = None,
        solve: Callable[[Simulation], Solution] = None,
        postprocess: Callable[[Solution], Solution] = None,
        output: Callable[[Solution], None] = None,
        validate: Callable[[Solution], bool] = None,
        ) -> Simulation:
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

        sim.solutions[0] = _run_one_step(sim=sim, solve=solve, postprocess=postprocess, output=output, validate=validate)

        return sim

    #
    starttime = sim.solutions[1].time

    first_time_to_solve = sim.solutions[0].time

    timestep_size = first_time_to_solve - starttime

    stepcount = 0

    time = starttime

    while time <= (endtime + ENDTIME_TOLERANCE):

        if stepcount > 0:

            sim.solutions = rotate_deque(sim.solutions, 1)

            sim.solutions[0].function.assign(sim.solutions[1].function)

            sim.solutions[0].time = sim.solutions[1].time + timestep_size

            sim.solutions[0].checkpoint_index = sim.solutions[1].checkpoint_index + 1

        sim.solutions[0] = _run_one_step(sim=sim, solve=solve, postprocess=postprocess, output=output, validate=validate)

        print("Solved at time t = {}".format(time))

        stepcount += 1

        time = starttime + stepcount*timestep_size

    return sim


def rotate_deque(deque_to_rotate: Deque, n: int) -> Deque:
    """ Rotate a deque without losing type info """
    return deque_to_rotate.rotate(n)


def _run_one_step(
        sim: Simulation,
        solve: Callable[[Simulation], Solution],
        postprocess: Callable[[Solution], Solution] = None,
        output: Callable[[Solution], None] = None,
        validate: Callable[[Solution], bool] = None,
        ) -> Solution:

    solution = solve(sim)

    if postprocess:

        print("Postprocessing solution data")

        solution = postprocess(solution)

    if output:

        print("Writing outputs")

        output(solution)

    if validate:

        print("Validating latest solution data")

        valid = validate(solution)

        if not valid:

            raise Exception("Solution data failed validation")

    return solution
