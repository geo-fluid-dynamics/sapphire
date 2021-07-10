"""Enthalpy-porosity module

Use this for convection-coupled melting and solidification of pure materials (i.e. without solutes).

Dirichlet BC's must not be placed on the pressure.
The returned pressure solution will always have zero mean.

Non-homogeneous Neumann BC's are not implemented.
"""
from typing import Callable, Tuple, Dict, Any
from sapphire.time_discretization import ufl_time_discrete_terms
from sapphire.data.solution import Solution
from sapphire.time_discretization import bdf
from sapphire.forms.natural_convection import linear_boussinesq_buoyancy, mass_residual, element, COMPONENT_NAMES  # pylint: disable=unused-import
from sapphire.forms.natural_convection import momentum_residual as natural_convection_momentum_residual
from firedrake import dot, grad, div, erf, sqrt, assemble, dx, interpolate


def porosity(solution: Solution):

    T = solution.ufl_fields.T

    T_m = solution.ufl_constants.melting_temperature

    sigma = solution.ufl_constants.porosity_smoothing_factor

    return 0.5*(1. + erf((T - T_m)/(sigma*sqrt(2))))


def phase_dependent_material_property(solid_to_liquid_ratio: float) -> Callable:

    p_sl = solid_to_liquid_ratio

    if p_sl.__float__() < 0:

        raise Exception("The solid-to-liquid ratio for any phase dependent material property must be non-negative.")

    def p(liquid_volume_fraction):

        f_l = liquid_volume_fraction

        return p_sl + (1 - p_sl)*f_l

    return p


def volumetric_heat_capacity(solution: Solution):

    rho_sl = solution.ufl_constants.density_solid_to_liquid_ratio

    c_sl = solution.ufl_constants.heat_capacity_solid_to_liquid_ratio

    phi = porosity(solution)

    return phase_dependent_material_property(rho_sl*c_sl)(phi)


def thermal_conductivity(solution: Solution):

    k_sl = solution.ufl_constants.thermal_conductivity_solid_to_liquid_ratio

    phi = porosity(solution)

    return phase_dependent_material_property(k_sl)(phi)


def solid_velocity_relaxation(solution: Solution):

    phi = porosity(solution)

    tau = solution.ufl_constants.solid_velocity_relaxation_factor

    return 1/tau*(1 - phi)


def momentum_residual(solutions: Tuple[Solution], buoyancy: Callable[[Solution], Any]) -> Any:
    """Momentum residual for natural convection governed by the Navier-Stokes-Boussinesq equations.

    Non-homogeneous Neumann BC's are not implemented for the velocity.
    """
    solution = solutions[0]

    U = solution.ufl_fields.U

    psi_U = solution.test_functions.U

    U_t = ufl_time_discrete_terms(solutions).U_t  # pylint: disable=no-member

    d = solid_velocity_relaxation(solution)

    _dx = dx(degree=solution.quadrature_degree)

    r = natural_convection_momentum_residual(solutions, buoyancy=buoyancy)

    return r + dot(psi_U, U_t + d*U)*_dx


def energy_residual(solutions: Tuple[Solution]) -> Any:
    """Energy residual formulated as convection and diffusion of a temperature field"""

    solution = solutions[0]

    CT_t = bdf(tuple(volumetric_heat_capacity(sol)*sol.ufl_fields.T for sol in solutions), solution.ufl_constants.timestep_size)

    phi_t = bdf(tuple(porosity(sol) for sol in solutions), solution.ufl_constants.timestep_size)

    Re = solution.ufl_constants.reynolds_number

    Pr = solution.ufl_constants.prandtl_number

    Ste = solution.ufl_constants.stefan_number

    U = solution.ufl_fields.U

    T = solution.ufl_fields.T

    C = volumetric_heat_capacity(solution)

    k = thermal_conductivity(solution)

    psi_T = solution.test_functions.T

    _dx = dx(degree=solution.quadrature_degree)

    return (psi_T*(CT_t + 1/Ste*phi_t + dot(U, grad(C*T))) + 1/(Re*Pr)*dot(grad(psi_T), k*grad(T)))*_dx


def residual(solutions: Tuple[Solution], buoyancy: Callable[[Solution], Any] = None) -> Any:
    """Sum of the mass, momentum, and energy residuals"""
    if buoyancy is None:

        buoyancy = linear_boussinesq_buoyancy

    return mass_residual(solutions) + momentum_residual(solutions, buoyancy=buoyancy) + energy_residual(solutions)


def postprocess(solution: Solution) -> Dict:

    phi = porosity(solution)

    _dx = dx(degree=solution.quadrature_degree)

    return {
        '\\phi': interpolate(phi, solution.subfunctions.T.function_space()),
        'liquid_area': assemble(phi*_dx),
        'velocity_divergence': assemble(div(solution.ufl_fields.U)*_dx)}
