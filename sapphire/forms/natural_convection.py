"""Natural convection module

Dirichlet BC's should not be placed on the pressure.

Non-homogeneous Neumann BC's are not implemented.
"""
from typing import Callable, Tuple, Any
from sapphire.data.solution import Solution
from sapphire.forms.incompressible_flow import mass_residual
from sapphire.forms.incompressible_flow import momentum_residual as incompressible_flow_momentum_residual
from firedrake import Cell, MixedElement, FiniteElement, VectorElement, Constant, dot, grad, dx


COMPONENT_NAMES = ('p', 'U', 'T')


def element(cell: Cell, taylor_hood_velocity_degree: int, temperature_degree: int):

    if taylor_hood_velocity_degree < 2:

        raise Exception("Taylor-Hood velocity element degree must be at least 2")

    if temperature_degree < 1:

        raise Exception("Temperature element degree must be at least 1 because continuous Galerkin discretization is assumed")

    return MixedElement(
        FiniteElement('P', cell, taylor_hood_velocity_degree - 1),
        VectorElement('P', cell, taylor_hood_velocity_degree),
        FiniteElement('P', cell, temperature_degree))


def linear_boussinesq_buoyancy(solution: Solution) -> Any:

    T = solution.ufl_fields.T

    Re = solution.ufl_constants.reynolds_number

    Ra = solution.ufl_constants.rayleigh_number

    Pr = solution.ufl_constants.prandtl_number

    ghat = Constant(-solution.unit_vectors[1])

    return Ra/(Pr*Re**2)*T*ghat


def momentum_residual(solutions: Tuple[Solution], buoyancy: Callable[[Solution], Any] = None) -> Any:
    """Momentum residual for natural convection governed by the Navier-Stokes-Boussinesq equations.

    Non-homogeneous Neumann BC's are not implemented for the velocity.
    """
    solution = solutions[0]

    psi_U = solution.test_functions.U

    b = buoyancy(solution)

    _dx = dx(degree=solution.quadrature_degree)

    return incompressible_flow_momentum_residual(solutions) + dot(psi_U, b)*_dx


def energy_residual(solutions: Tuple[Solution]) -> Any:
    """Energy residual formulated as convection and diffusion of a temperature field"""
    solution = solutions[0]

    Re = solution.ufl_constants.reynolds_number

    Pr = solution.ufl_constants.prandtl_number

    U = solution.ufl_fields.U

    T = solution.ufl_fields.T

    psi_T = solution.test_functions.T

    _dx = dx(degree=solution.quadrature_degree)

    return (psi_T*dot(U, grad(T)) + dot(grad(psi_T), 1./(Re*Pr)*grad(T)))*_dx


def residual(solutions: Tuple[Solution], buoyancy: Callable[[Solution], Any] = None) -> Any:
    """Sum of the mass, momentum, and energy residuals"""
    if buoyancy is None:

        buoyancy = linear_boussinesq_buoyancy

    return mass_residual(solutions) + momentum_residual(solutions, buoyancy=buoyancy) + energy_residual(solutions)
