"""Incompressible flow module

Dirichlet BC's should not be placed on the pressure.

Non-homogeneous Neumann BC's are not implemented.
"""
from typing import Tuple, Any
from sapphire.data.solution import Solution
from firedrake import Cell, MixedElement, FiniteElement, VectorElement, inner, dot, grad, div, sym, dx


COMPONENT_NAMES = ('p', 'U')


def element(cell: Cell, taylor_hood_velocity_degree: int):

    if taylor_hood_velocity_degree < 2:

        raise Exception("Taylor-Hood velocity element degree must be at least 2")

    return MixedElement(
        FiniteElement('P', cell, taylor_hood_velocity_degree - 1),
        VectorElement('P', cell, taylor_hood_velocity_degree))


def mass_residual(solutions: Tuple[Solution]) -> Any:
    """Mass residual assuming incompressible flow"""
    solution = solutions[0]

    U = solution.ufl_fields.U

    psi_p = solution.test_functions.p

    _dx = dx(degree=solution.quadrature_degree)

    return psi_p*div(U)*_dx


def momentum_residual(solutions: Tuple[Solution]) -> Any:
    """Momentum residual for natural convection governed by the Navier-Stokes-Boussinesq equations.

    Non-homogeneous Neumann BC's are not implemented for the velocity.
    """
    solution = solutions[0]

    p = solution.ufl_fields.p

    U = solution.ufl_fields.U

    psi_U = solution.test_functions.U

    Re = solution.ufl_constants.reynolds_number

    _dx = dx(degree=solution.quadrature_degree)

    return (dot(psi_U, grad(U)*U) - div(psi_U)*p + 2./Re*inner(sym(grad(psi_U)), sym(grad(U))))*_dx


def residual(solutions: Tuple[Solution]) -> Any:
    """Sum of the mass and momentum residuals"""

    return mass_residual(solutions) + momentum_residual(solutions)
