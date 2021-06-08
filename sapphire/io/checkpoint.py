""" Checkpointing module """
from sapphire.data import Simulation
import firedrake as fe


def write_checkpoint(sim: Simulation, file_basename: str):
    """Write checkpoint for restarting and/or post-processing.

    A solution is stored for each state in states.

    Restarting requires states filling the time discretization stencil.

    If a checkpoint already exists for a state's time, then no new checkpoint is written for that state.
    """

    checkpointer = fe.DumbCheckpoint(basename=file_basename, mode=fe.FILE_UPDATE)

    stored_times, _ = checkpointer.get_timesteps()

    for solution in sim.solutions:

        if solution.time in stored_times:

            continue

        checkpointer.set_timestep(t=solution.time, idx=solution.checkpoint_index)

        print("Writing checkpoint to {}".format(checkpointer.h5file.filename))

        checkpointer.store(solution.function, name="solution.function")

        for key in solution.post_processed_functions.keys():

            checkpointer.store(solution.post_processed_functions[key], name="solution.post_processed_functions."+key)


def read_checkpoint(sim: Simulation, file_basename: str) -> Simulation:

    checkpointer = fe.DumbCheckpoint(basename=file_basename, mode=fe.FILE_READ)

    stored_times, _ = checkpointer.get_timesteps()

    for solution in sim.solutions:

        assert(solution.time in stored_times)

        checkpointer.set_timestep(t=solution.time, idx=solution.checkpoint_index)

        print("Reading checkpoint from {}".format(checkpointer.h5file.filename))

        checkpointer.load(solution.function, name="solution.function")

    return sim
