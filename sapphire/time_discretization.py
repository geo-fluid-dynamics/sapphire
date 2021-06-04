"""Time discretization formulas

Derived from

    @book{ascher1998computer,
      title={Computer methods for ordinary differential equations and differential-algebraic equations},
      author={Ascher, Uri M and Petzold, Linda R},
      volume={61},
      year={1998},
      publisher={Siam}
    }
"""
from typing import Deque, Any
from sapphire.data.solution import Solution
from firedrake import Constant, split


def ufl_time_discrete_terms(
        solutions: Deque[Solution],
        timestep_size: Constant
        ) -> Any:
    """Returns backward difference time discretization.

    The backward difference formula's stencil size is determine by the number of solutions provided, i.e. `len(solutions)`.
    For example, if `len(solutions == 3)`, then the second-order BDF2 method will be used, because it involves solutions at three discrete times.

    The return type depends on whether or not the solution is based on a mixed finite element.
    For mixed finite elements, a list of time discrete terms will be returned, each item corresponding to one of the sub-elements of the mixed element.
    Otherwise, a single term will be returned.


    More details:

    The return type design choice was made, rather than always returning a list (e.g. with only one item if not using a mixed element), so that it would be more intuitive when not using mixed elements.

    This implementation assumes constant time step size.
    Variable time step sizes change the BDF formula for all except first order.
    """
    functions = [solutions[i].function for i in range(len(solutions))]

    component_count = len(functions[0].split())

    return [
        bdf(
            [split(functions[j])[i] for j in range(len(functions))],
            timestep_size=timestep_size)
        for i in range(component_count)]


def bdf(solutions, timestep_size):
    """ Backward difference formulas with constant time step size """
    order = len(solutions) - 1

    if order < 1:

        raise ValueError("At least two solutions are needed for the minimum time discretization stencil.")

    # Table of BDF method coefficients
    if order == 1:

        alphas = (1., -1.)

    elif order == 2:

        alphas = (3./2., -2., 1./2.)

    elif order == 3:

        alphas = (11./6., -3., 3./2., -1./3.)

    elif order == 4:

        alphas = (25./12., -4., 3., -4./3., 1./4.)

    elif order == 5:

        alphas = (137./60., -5., 5., -10./3., 5./4., -1./5.)

    elif order == 6:

        alphas = (147./60., -6., 15./2., -20./3., 15./4., -6./5., 1./6.)

    else:

        raise ValueError("BDF is not zero-stable with order > 6.")

    u_t = alphas[-1]*solutions[-1]

    for alpha, u in zip(alphas[:-1], solutions[:-1]):

        u_t += alpha*u

    u_t /= timestep_size

    return u_t
