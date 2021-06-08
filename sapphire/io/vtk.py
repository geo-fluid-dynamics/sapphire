import firedrake as fe


def write_solution_to_vtk(
        sim,
        solution=None,
        dependent_functions=None,
        time=None,
        file=None):

    if solution is None:

        solution = sim.solution

    if time is None:

        time = sim.time

    if file is None:

        file = sim.solution_file

    functions_to_write = solution.split()

    if dependent_functions is None:

        if hasattr(sim, "postprocessed_functions"):

            dependent_functions = sim.postprocessed_functions

    if dependent_functions is not None:

        functions_to_write += dependent_functions

    print("Writing solution to {}".format(file.filename))

    if time is None:

        file.write(functions_to_write)

    else:

        if isinstance(time, float):

            timefloat = time

        elif isinstance(time, fe.Constant):

            timefloat = time.__float__()

        file.write(*functions_to_write, time=timefloat)
