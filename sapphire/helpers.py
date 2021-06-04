"""This module provides helper functions which are independent of other `sapphire` modules"""
import typing
import numpy as np
import firedrake as fe


def rotate_deque(deque_to_rotate: typing.Deque, n: int) -> typing.Deque:
    """ Rotate a deque without losing type info """
    return deque_to_rotate.rotate(n)


def mesh(_function_space: fe.FunctionSpace) -> fe.Mesh:
    """@todo Trying to retain type info"""
    return _function_space.mesh()


def element(_function_space: fe.FunctionSpace) -> typing.Union[fe.FiniteElement, fe.VectorElement, fe.MixedElement]:
    """Get element without losing type info"""
    return _function_space.ufl_element()


def function_space(_function: fe.Function) -> fe.FunctionSpace:
    """@todo Trying to retain type info"""
    return _function.function_space()


def assign_function_values(from_function: fe.Function, to_function: fe.Function) -> fe.Function:
    """ firedrake.Function.assign without losing type info """
    to_function = to_function.assign(from_function)

    return to_function


def assign_constant(constant: fe.Constant, value: typing.Union[float, typing.Tuple[float]]) -> fe.Constant:

    return constant.assign(value)


def snes_iteration_count(solver: fe.NonlinearVariationalSolver) -> int:
    """ Get iteration number without losing type info """
    return solver.snes.getIterationNumber()


def magnitude(vector: typing.Iterable) -> float:

    return np.sqrt(np.sum(np.square(vector)))


def normalize_to_unit_vector(vector: np.array) -> typing.Tuple[float]:

    return tuple(vector/magnitude(vector))


def verify_function_values_at_coordinates(
        function: fe.Function,
        coordinates: typing.Union[typing.Tuple[float], typing.Tuple[typing.Tuple[float]]],
        expected_values: typing.Union[typing.Tuple[float], typing.Tuple[typing.Tuple[float]]],
        absolute_tolerances: typing.Union[typing.Tuple[float], typing.Tuple[typing.Tuple[float]]]):

    if len(expected_values) != len(coordinates) or (len(expected_values) != len(absolute_tolerances)):

        raise Exception("There must be an expected value and a tolerance for each coordinate.")

    for coordinate, expected_value, tolerance in zip(coordinates, expected_values, absolute_tolerances):

        value = function.at(coordinate)

        if isinstance(value, float):

            value = (value,)

            expected_value = (expected_value,)

        print("Expected {} and found {}.".format(expected_value, value))

        for i, v_i in enumerate(value):

            if expected_value[i] is None:

                continue

            error = abs(v_i - expected_value[i])

            if error > tolerance[i]:

                raise Exception("Absolute error ({}) is greater than tolerance ({})".format(error, tolerance[i]))
