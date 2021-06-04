"""This module provides helper functions which are independent of other `sapphire` modules"""
import typing
import firedrake as fe


def rotate_deque(deque_to_rotate: typing.Deque, n: int) -> typing.Deque:
    """ Rotate a deque without losing type info """
    return deque_to_rotate.rotate(n)


def mesh(_function_space: fe.FunctionSpace) -> fe.Mesh:
    """@todo Trying to retain type info"""
    return _function_space.mesh()


def geometric_dimension(_mesh: fe.Mesh) -> int:
    """Get geometric dimension without losing type info"""
    return _mesh.geometric_dimension()


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
