"""Plotting module"""
from pathlib import Path
from matplotlib import use as matplotlib_use
from matplotlib import pyplot as plt
from sapphire.data.solution import Solution
from firedrake import FiniteElement, VectorElement, Function, tricontourf, streamplot, quiver
from firedrake import plot as plot_1d


matplotlib_use('Agg')  # Only use back-end to prevent displaying image

DEFAULT_RESOLUTION = 256

DEFAULT_SAVEFIG_KWARGS = {
    'dpi': DEFAULT_RESOLUTION,
    'bbox_inches': "tight",
    'pad_inches': 0.02}

VECTOR_PLOT_FUNCTION_NAMES = ('quiver', 'streamplot')


def plot(
        solution: Solution,
        output_directory_path: str,
        vector_plot_function_name: str = 'quiver',
        savefig_kwargs: dict = None):
    """Save plots of each function (in non-interactive mode)"""
    Path(output_directory_path).mkdir(parents=True, exist_ok=True)

    if savefig_kwargs is None:

        savefig_kwargs = DEFAULT_SAVEFIG_KWARGS

    if vector_plot_function_name not in VECTOR_PLOT_FUNCTION_NAMES:

        raise Exception("Invalid `vector_plot_function_name` ({}) not in `VECTOR_PLOT_FUNCTION_NAMES` ({})".format(vector_plot_function_name, VECTOR_PLOT_FUNCTION_NAMES))

    functions = list(solution.subfunctions)

    names = list(solution.component_names)

    for key in solution.post_processed_objects:

        ppo = solution.post_processed_objects[key]

        if isinstance(ppo, Function):

            functions.append(ppo)

            names.append(key)

    for f, label in zip(functions, names):

        _, axes = plt.subplots()

        if solution.geometric_dimension == 1:

            plot_1d(f, axes=axes)

        elif solution.geometric_dimension == 2:

            element = f.function_space().ufl_element()

            if isinstance(element, VectorElement):

                if vector_plot_function_name == 'quiver':

                    mappable = quiver(f, axes=axes)

                elif vector_plot_function_name == 'streamplot':

                    if f.vector().array().max() == f.vector().array().min():
                        # `firedrake.streamplot` seems to not handle this case very well, raises `ValueError: Points to evaluate are inconsistent among processes.`
                        continue

                    print("Plotting streamlines. Computing them can take a while")

                    mappable = streamplot(f, axes=axes)

            elif isinstance(element, FiniteElement):

                mappable = tricontourf(f, axes=axes, levels=128)

            else:

                raise Exception("Unable to plot unexpected element type")

            plt.colorbar(mappable=mappable, ax=axes)

        else:

            raise Exception("Default plot function is only implemented for 1D and 2D geometries")

        title = "${}$".format(label)

        if solution.time is not None:

            title += ", $t = {}$".format(solution.time)

        plt.title(title)

        plt.xlabel('$x$')

        if solution.geometric_dimension == 2:

            plt.ylabel('$y$')

            axes.set_aspect('equal', 'box')

        filename = label

        if solution.time is not None:

            filename += '_it{}'.format(solution.checkpoint_index)

        filepath = output_directory_path + '/' + filename + '.png'

        print("Writing plot to {}".format(filepath))

        plt.savefig(filepath, **savefig_kwargs)

        plt.close()
