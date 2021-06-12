""" Checkpointing module """
from pathlib import Path
from sapphire.data.solution import Solution
from firedrake import DumbCheckpoint, Function, FILE_UPDATE, FILE_READ


def write_checkpoint(solution: Solution, filepath_without_extension: str):
    """Write checkpoint for restarting and/or post-processing.

    A solution is stored for each state in states.

    Restarting requires states filling the time discretization stencil.

    If a checkpoint already exists for a state's time, then no new checkpoint is written for that state.
    """
    Path(filepath_without_extension).parent.mkdir(parents=True, exist_ok=True)

    checkpointer = DumbCheckpoint(basename=filepath_without_extension, mode=FILE_UPDATE)

    checkpointer.set_timestep(t=solution.time, idx=solution.checkpoint_index)

    print("Writing checkpoint to {}".format(checkpointer.h5file.filename))

    checkpointer.store(solution.function, name='solution.function')

    for key in solution.post_processed_objects:

        ppo = solution.post_processed_objects[key]

        if isinstance(ppo, Function):

            checkpointer.store(ppo, name='solution.post_processed_objects.'+key)


def read_checkpoint(solution_function: Function, time: float, index: int, filepath_without_extension: str):

    checkpointer = DumbCheckpoint(basename=filepath_without_extension, mode=FILE_READ)

    stored_times, _ = checkpointer.get_timesteps()

    if time not in stored_times:

        raise Exception("Checkpoint file does not contain a solution for time {}. Stored times are {}".format(time, stored_times))

    checkpointer.set_timestep(t=time, idx=index)

    print("Reading checkpoint from {}".format(checkpointer.h5file.filename))

    checkpointer.load(solution_function, name='solution.function')
