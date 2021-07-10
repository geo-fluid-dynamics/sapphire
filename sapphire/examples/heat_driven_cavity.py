""" Example heat-driven cavity simulation

The result is compared to data published in

    @article{wang2010comprehensive,
        author = {Ghia, Urmila and Ghia, K.N and Shin, C.T},
        year = {1982},
        month = {12},
        pages = {387-411},
        title = {High-Re solutions for incompressible flow using the
        Navier-Stokes equations and a multigrid method},
        volume = {48},
        journal = {Journal of Computational Physics},
        doi = {10.1016/0021-9991(82)90058-4}
    }
"""
from typing import Tuple
from sapphire import Mesh, Solution, Simulation, solve_with_bounded_continuation_sequence, run, plot
from sapphire import solve as default_solve
from sapphire.forms.natural_convection import COMPONENT_NAMES, element, residual
from sapphire.examples.lid_driven_cavity import DEFAULT_FIREDRAKE_SOLVER_PARAMETERS, cavity_mesh
from sapphire.helpers.pointwise_verification import verify_function_values_at_points
from firedrake import UnitSquareMesh, DirichletBC, MixedVectorSpaceBasis, VectorSpaceBasis


def dirichlet_boundary_conditions(solution: Solution):
    """No-slip BCs for the velocity on every wall and constant temperature BCs on the left and right walls.

    The weak formulation does not admit Dirichlet boundary conditions on the pressure solution.
    To make the solution unique, the returned pressure solution will always be post-processed to have zero mean.
    """

    return (
        DirichletBC(solution.function_subspaces.U, (0,)*solution.geometric_dimension, "on_boundary"),
        DirichletBC(solution.function_subspaces.T, solution.ufl_constants.hotwall_temperature, solution.mesh.boundaries['left']),
        DirichletBC(solution.function_subspaces.T, solution.ufl_constants.coldwall_temperature, solution.mesh.boundaries['right']))


def nullspace(solution: Solution):
    """Inform solver that pressure solution is not unique.

    It is only defined up to adding an arbitrary constant because there will be no boundary conditions on the pressure.
    """
    return MixedVectorSpaceBasis(
        solution.function.function_space(),
        [VectorSpaceBasis(constant=True), solution.function_subspaces.U, solution.function_subspaces.T])


def solve_with_rayleigh_number_continuation(sim: Simulation):

    Ra = sim.solutions[0].ufl_constants.rayleigh_number

    solve_with_bounded_continuation_sequence(
        sim=sim,
        solve=default_solve,
        continuation_parameter_and_name=(Ra, 'Ra'),
        initial_sequence=(1, Ra.__float__()))


def output(solution: Solution):

    plot(solution=solution, output_directory_path="sapphire_output/heat_driven_cavity/plots/")


def run_simulation(
        reynolds_number=1.,
        rayleigh_number=1.e6,
        prandtl_number=0.71,
        hotwall_temperature=0.5,
        coldwall_temperature=-0.5,
        mesh_dimensions=(20, 20),
        taylor_hood_velocity_element_degree=2,
        temperature_element_degree=2,
        firedrake_solver_parameters=None):

    if firedrake_solver_parameters is None:

        firedrake_solver_parameters = DEFAULT_FIREDRAKE_SOLVER_PARAMETERS

    _mesh = cavity_mesh(nx=mesh_dimensions[0], ny=mesh_dimensions[1], Lx=1, Ly=1)

    sim = Simulation(
        mesh=_mesh,
        element=element(_mesh.cell, taylor_hood_velocity_element_degree, temperature_element_degree),
        solution_component_names=COMPONENT_NAMES,
        ufl_constants={
            'reynolds_number': reynolds_number,
            'rayleigh_number': rayleigh_number,
            'prandtl_number': prandtl_number,
            'hotwall_temperature': hotwall_temperature,
            'coldwall_temperature': coldwall_temperature},
        residual=residual,
        dirichlet_boundary_conditions=dirichlet_boundary_conditions,
        nullspace=nullspace,
        firedrake_solver_parameters=firedrake_solver_parameters,
        time_discretization_stencil_size=1)

    return run(sim=sim, solve=solve_with_rayleigh_number_continuation, output=output)


def verify_default_simulation():

    sim = run_simulation()

    solution = sim.solutions[0]

    Ra = solution.ufl_constants.rayleigh_number.__float__()

    Pr = solution.ufl_constants.prandtl_number.__float__()

    verify_function_values_at_points(
        function=sim.solutions[0].subfunctions.U,
        points=[(0.5, y) for y in (0., 0.15, 0.34999, 0.5, 0.65, 0.84999)],
        # Checking y coordinates 0.3499 and 0.8499 instead of 0.35, 0.85 because the `firedrake.Function` evaluation fails at the exact coordinates. See https://github.com/firedrakeproject/firedrake/issues/1340
        expected_values=[(U_x*Ra**0.5/Pr, None) for U_x in (0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649)],
        absolute_tolerances=[(tol*Ra**0.5/Pr, None) for tol in (1.e-12, 0.001, 0.001, 1.e-12, 0.001, 0.001)])


if __name__ == '__main__':

    verify_default_simulation()
