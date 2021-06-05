"""Natural convection module"""
from typing import Callable, Tuple, Any
from sapphire.data.solution import Solution
from firedrake import Cell, MixedElement, FiniteElement, VectorElement, Constant, inner, dot, grad, div, sym, dx


SOLUTION_FUNCTION_COMPONENT_NAMES = ('p', 'u', 'T')


def element(cell: Cell, taylor_hood_pressure_degree: int = 1, temperature_degree: int = 2) -> MixedElement:

    if taylor_hood_pressure_degree < 1:

        raise Exception("Taylor-Hood pressure element degree must be at least 1")

    if temperature_degree < 1:

        raise Exception("Temperature element degree must be at least 1 because continuous Galerkin discretization is assumed")

    return MixedElement(
        FiniteElement('P', cell, taylor_hood_pressure_degree),
        VectorElement('P', cell, taylor_hood_pressure_degree + 1),
        FiniteElement('P', cell, temperature_degree))


def linear_boussinesq_buoyancy(solution: Solution) -> Any:

    T = solution.ufl_fields.T

    Re = solution.ufl_constants.reynolds_number

    Ra = solution.ufl_constants.rayleigh_number

    Pr = solution.ufl_constants.prandtl_number

    ghat = Constant(-solution.unit_vectors[1])

    return Ra/(Pr*Re**2)*T*ghat


def mass_residual(solutions: Tuple[Solution]) -> Any:
    """Mass residual assuming incompressible flow"""
    solution = solutions[0]

    u = solution.ufl_fields.u

    psi_p = solution.test_functions.p

    _dx = dx(degree=solution.quadrature_degree)

    return psi_p*div(u)*_dx


def momentum_residual(solutions: Tuple[Solution], buoyancy: Callable[[Solution], Any] = None) -> Any:
    """Momentum residual for natural convection governed by the Navier-Stokes-Boussinesq equations.

    Non-homogeneous Neumann BC's are not implemented for the velocity.
    """
    solution = solutions[0]

    p = solution.ufl_fields.p

    u = solution.ufl_fields.u

    psi_u = solution.test_functions.u

    b = buoyancy(solution)

    Re = solution.ufl_constants.reynolds_number

    _dx = dx(degree=solution.quadrature_degree)

    return (dot(psi_u, grad(u)*u + b) - div(psi_u)*p + 2./Re*inner(sym(grad(psi_u)), sym(grad(u))))*_dx


def energy_residual(solutions: Tuple[Solution]) -> Any:
    """Energy residual formulated as convection and diffusion of a temperature field"""
    solution = solutions[0]

    Re = solution.ufl_constants.reynolds_number

    Pr = solution.ufl_constants.prandtl_number

    u = solution.ufl_fields.u

    T = solution.ufl_fields.T

    psi_T = solution.test_functions.T

    _dx = dx(degree=solution.quadrature_degree)

    return (psi_T*dot(u, grad(T)) + dot(grad(psi_T), 1./(Re*Pr)*grad(T)))*_dx


def residual(solutions: Tuple[Solution], buoyancy: Callable[[Solution], Any] = None) -> Any:
    """Sum of the mass, momentum, and energy residuals"""
    if buoyancy is None:

        buoyancy = linear_boussinesq_buoyancy

    return mass_residual(solutions) + momentum_residual(solutions, buoyancy=buoyancy) + energy_residual(solutions)
