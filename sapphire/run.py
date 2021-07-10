"""Module for running simulations"""
from logging import warning
from typing import Callable, Dict, Any
from sapphire.data.solution import Solution
from sapphire.data.simulation import Simulation
from sapphire.solve import solve as default_solve
from sapphire.io.plot import plot


ENDTIME_TOLERANCE = 1.e-8
""" Allows endtime to be only approximately reached.

This is larger than a typical floating point comparison tolerance because errors accumulate between timesteps.
"""


def default_output(solution: Solution, output_directory_path: str):

    plot(solution=solution, output_directory_path=output_directory_path)


def run(  # pylint: disable=too-many-arguments
        sim: Simulation,
        endtime: float = None,
        solve: Callable[[Simulation], None] = None,
        postprocess: Callable[[Solution], Dict[str, Any]] = None,
        output: Callable[[Solution], None] = None,
        validate: Callable[[Solution], None] = None,
        ):
    """Run simulation forward in time.

    :param sim: The simulation data.

    :param endtime: Run the simulation until reaching this time. Use default value None for steady state problems.

    :param solve: This is called to solve each time step.

    :param postprocess: This is called to post-process the simulation data before writing outputs.

    :param output: This is called to write outputs after solving each time step.

    :param validate: This is called to validate data before solving.

    Modifies `sim.solutions`.
    """
    time = sim.solutions[0].time

    if (time is not None) and (time >= (endtime - ENDTIME_TOLERANCE)):

        warning("End time (t_f = {}) already reached (t = {})".format(endtime, time))

        return

    for function, default_function, name in zip((solve, output), (default_solve, default_output), ('solve', 'output')):

        if function is None:

            function = default_function

            warning("`run` is using default `{}` function".format(name))

    if validate is not None:

        print("Validating all initial solution data")

        for solution in sim.solutions:

            validate(solution)

    if time is None:

        if endtime is not None:

            raise Exception("`endtime` was specified but `solution.time` is None")

        if len(sim.solutions) > 1:

            raise Exception("`sim.solutions` should only contain a single solution if there is no time discretization")

        _run_one_step(sim=sim, solve=solve, postprocess=postprocess, output=output, validate=validate)

        return sim

    #
    while time < (endtime - ENDTIME_TOLERANCE):

        timestep_size = sim.solutions[0].ufl_constants.timestep_size.__float__()

        time += timestep_size

        sim.solutions.rotate(1)

        copy_solution_values(sim.solutions[1], sim.solutions[0])

        sim.solutions[0].time = time

        sim.solutions[0].checkpoint_index = sim.solutions[1].checkpoint_index + 1

        _run_one_step(sim=sim, solve=solve, postprocess=postprocess, output=output, validate=validate)

        print("Solved at time t = {}".format(sim.solutions[0].time))


def copy_solution_values(from_solution: Solution, to_solution: Solution):

    to_solution.function.assign(from_solution.function)

    for cname in from_solution.ufl_constants._fields:

        getattr(to_solution.ufl_constants, cname).assign(getattr(from_solution.ufl_constants, cname))

    for key in from_solution.extras.keys():

        to_solution.extras[key] = from_solution.extras[key]


def _run_one_step(
        sim: Simulation,
        solve: Callable[[Simulation], None],
        postprocess: Callable[[Solution], Dict[str, Any]] = None,
        output: Callable[[Solution], None] = None,
        validate: Callable[[Solution], None] = None,
        ):

    solve(sim)

    solution = sim.solutions[0]

    if postprocess is not None:

        print("Postprocessing solution data")

        solution.post_processed_objects = postprocess(solution)

    if output is not None:

        print("Writing outputs")

        output(solution)

    if validate is not None:

        print("Validating latest solution data")

        validate(solution)
