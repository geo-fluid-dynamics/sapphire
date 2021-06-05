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
from typing import Tuple, Union, Callable, Dict, Any
from sapphire import Problem, Solver, Solution, Simulation, solve_with_bounded_continuation_sequence, run, plot
from sapphire import solve as default_solve
from sapphire.forms.natural_convection import SOLUTION_FUNCTION_COMPONENT_NAMES, element, residual
from sapphire.forms.natural_convection import linear_boussinesq_buoyancy as default_buoyancy
from sapphire.helpers.pointwise_verification import verify_function_values_at_points
from firedrake import UnitSquareMesh, UnitCubeMesh, Function, FunctionSpace, DirichletBC, MixedVectorSpaceBasis, VectorSpaceBasis, dx, assemble


MESH_HOTWALL_ID = 1

MESH_COLDWALL_ID = 2

DEFAULT_MESH_DIMENSIONS = (20, 20)


def default_mesh(mesh_dimensions: Union[Tuple[int, int], Tuple[int, int, int]] = None):

    if mesh_dimensions is None:

        mesh_dimensions = DEFAULT_MESH_DIMENSIONS

    if len(mesh_dimensions) == 2:

        return UnitSquareMesh(*mesh_dimensions)

    elif len(mesh_dimensions) == 3:

        return UnitCubeMesh(*mesh_dimensions)


def dirichlet_boundary_conditions(solution: Solution):
    """No-slip BCs for the velocity on every wall and constant temperature BCs on the left and right walls.

    The weak formulation does not admit Dirichlet boundary conditions on the pressure solution.
    To make the solution unique, the returned pressure solution will always be post-processed to have zero mean.
    """

    return (
        DirichletBC(solution.function_subspaces.u, (0,)*solution.geometric_dimension, "on_boundary"),
        DirichletBC(solution.function_subspaces.T, solution.ufl_constants.hotwall_temperature, MESH_HOTWALL_ID),
        DirichletBC(solution.function_subspaces.T, solution.ufl_constants.coldwall_temperature, MESH_COLDWALL_ID))


def nullspace(solution: Solution):
    """Inform solver that pressure solution is not unique.

    It is only defined up to adding an arbitrary constant because there will be no boundary conditions on the pressure.
    """
    return MixedVectorSpaceBasis(
        solution.function_space,
        [VectorSpaceBasis(constant=True), solution.function_subspaces.u, solution.function_subspaces.T])


def solve_and_subtract_mean_pressure(sim: Simulation):

    solution = default_solve(sim)

    print("Subtracting mean pressure to make solution unique")

    p = solution.ufl_fields.p

    mean_pressure = assemble(p*dx(degree=solution.quadrature_degree))

    p = solution.subfunctions.p

    p.assign(p - mean_pressure)

    print("Done subtracting mean pressure")

    return solution


def solve_with_rayleigh_number_continuation(sim: Simulation):

    Ra = sim.solutions[0].ufl_constants.rayleigh_number

    return solve_with_bounded_continuation_sequence(
        sim=sim,
        solve=default_solve,
        continuation_parameter_and_name=(Ra, 'Ra'),
        initial_sequence=(1, Ra.__float__()))


def solve(sim: Simulation):

    return solve_with_rayleigh_number_continuation(sim)


def output(solution: Solution, outdir_path: str = "sapphire_output/heat_driven_cavity/"):

    plot(
        solution=solution,
        outdir_path=outdir_path)


def simulation(
        ufl_constants: Dict[str, float],
        buoyancy: Callable[[Solution], Any] = None,
        mesh: Any = None,
        taylor_hood_pressure_element_degree=1,
        temperature_element_degree=2):

    if buoyancy is None:

        buoyancy = default_buoyancy

    if mesh is None:

        mesh = default_mesh()

    solution = Solution(
        function=Function(FunctionSpace(
            mesh,
            element(mesh.ufl_cell(), taylor_hood_pressure_element_degree, temperature_element_degree))),
        function_component_names=SOLUTION_FUNCTION_COMPONENT_NAMES,
        ufl_constants=ufl_constants)

    def _residual(_solution):

        return residual(_solution, buoyancy=buoyancy)

    problem = Problem(
        residual=_residual,
        dirichlet_boundary_conditions=dirichlet_boundary_conditions(solution))

    solver = Solver(nullspace=nullspace(solution))

    sim = Simulation(solutions=(solution,), problem=problem, solver=solver)

    return sim


def run_simulation(
        reynolds_number=1.,
        rayleigh_number=1.e6,
        prandtl_number=0.71,
        hotwall_temperature=0.5,
        coldwall_temperature=-0.5,
        mesh_dimensions=(20, 20),
        taylor_hood_pressure_element_degree=1,
        temperature_element_degree=2):

    sim = simulation(
        ufl_constants={
            'reynolds_number': reynolds_number,
            'rayleigh_number': rayleigh_number,
            'prandtl_number': prandtl_number,
            'hotwall_temperature': hotwall_temperature,
            'coldwall_temperature': coldwall_temperature},
        mesh=default_mesh(mesh_dimensions),
        taylor_hood_pressure_element_degree=taylor_hood_pressure_element_degree,
        temperature_element_degree=temperature_element_degree)

    return run(sim=sim, solve=solve, output=output)


def verify_default_simulation():

    sim = run_simulation()

    solution = sim.solutions[0]

    Ra = solution.ufl_constants.rayleigh_number.__float__()

    Pr = solution.ufl_constants.prandtl_number.__float__()

    verify_function_values_at_points(
        function=sim.solutions[0].subfunctions.u,
        points=[(0.5, y) for y in (0., 0.15, 0.34999, 0.5, 0.65, 0.84999)],
        # Checking y coordinates 0.3499 and 0.8499 instead of 0.35, 0.85 because the `firedrake.Function` evaluation fails at the exact coordinates. See https://github.com/firedrakeproject/firedrake/issues/1340
        expected_values=[(u_x*Ra**0.5/Pr, None) for u_x in (0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649)],
        absolute_tolerances=[(tol*Ra**0.5/Pr, None) for tol in (1.e-12, 0.001, 0.001, 1.e-12, 0.001, 0.001)])


if __name__ == '__main__':

    verify_default_simulation()
