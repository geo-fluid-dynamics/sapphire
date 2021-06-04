""" Example heat-driven cavity simulation governed by the Navier-Stokes-Boussinesq equations

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
import typing
import sapphire
import firedrake as fe


SOLUTION_FUNCTION_COMPONENT_NAMES = ('p', 'u', 'T')

MESH_HOTWALL_ID = 1

MESH_COLDWALL_ID = 2


def mesh(mesh_dimensions: typing.Tuple[int, int]) -> fe.Mesh:

    return fe.UnitSquareMesh(*mesh_dimensions)


def element(cell: fe.Cell, taylor_hood_pressure_degree: int = 1, temperature_degree: int = 2) -> fe.MixedElement:

    if taylor_hood_pressure_degree < 1:

        raise Exception("Taylor-Hood pressure element degree must be at least 1")

    if temperature_degree < 1:

        raise Exception("Temperature element degree must be at least 1 because continuous Galerkin discretization is assumed")

    return fe.MixedElement(
        fe.FiniteElement('P', cell, taylor_hood_pressure_degree),
        fe.VectorElement('P', cell, taylor_hood_pressure_degree + 1),
        fe.FiniteElement('P', cell, temperature_degree))


def nullspace(solution: sapphire.Solution) -> fe.MixedVectorSpaceBasis:
    """Inform solver that pressure solution is not unique.

    It is only defined up to adding an arbitrary constant because there will be no boundary conditions on the pressure.
    """
    return fe.MixedVectorSpaceBasis(
        solution.function_space,
        [fe.VectorSpaceBasis(constant=True),
         solution.function_subspaces.u,
         solution.function_subspaces.T])


def ufl_constants(
        hotwall_temperature: float,
        coldwall_temperature: float,
        reynolds_number: float,
        rayleigh_number: float,
        prandtl_number: float):

    return {
        'hotwall_temperature': hotwall_temperature,
        'coldwall_temperature': coldwall_temperature,
        'reynolds_number': reynolds_number,
        'rayleigh_number': rayleigh_number,
        'prandtl_number': prandtl_number}


def linear_boussinesq_buoyancy(solution: sapphire.Solution):

    T = solution.ufl_fields.T

    Re = solution.ufl_constants.reynolds_number

    Ra = solution.ufl_constants.rayleigh_number

    Pr = solution.ufl_constants.prandtl_number

    ghat = fe.Constant(-solution.unit_vectors[1])

    return Ra/(Pr*Re**2)*T*ghat


inner, dot, grad, div, sym = fe.inner, fe.dot, fe.grad, fe.div, fe.sym


def mass_residual(solution: sapphire.Solution):
    """Mass residual assuming incompressible flow"""
    u = solution.ufl_fields.u

    psi_p = solution.test_functions.p

    dx = fe.dx(degree=solution.quadrature_degree)

    return psi_p*div(u)*dx


def momentum_residual(solution: sapphire.Solution, buoyancy: typing.Callable[[sapphire.Solution], typing.Any] = None):
    """Momentum residual for natural convection governed by the Navier-Stokes-Boussinesq equations.

    Non-homogeneous Neumann BC's are not implemented for the velocity.
    """
    p = solution.ufl_fields.p

    u = solution.ufl_fields.u

    psi_u = solution.test_functions.u

    b = buoyancy(solution)

    Re = solution.ufl_constants.reynolds_number

    dx = fe.dx(degree=solution.quadrature_degree)

    return (dot(psi_u, grad(u)*u + b) - div(psi_u)*p + 2./Re*inner(sym(grad(psi_u)), sym(grad(u))))*dx


def energy_residual(solution: sapphire.Solution):
    """Energy residual formulated as convection and diffusion of a temperature field"""
    Re = solution.ufl_constants.reynolds_number

    Pr = solution.ufl_constants.prandtl_number

    u = solution.ufl_fields.u

    T = solution.ufl_fields.T

    psi_T = solution.test_functions.T

    dx = fe.dx(degree=solution.quadrature_degree)

    return (psi_T*dot(u, grad(T)) + dot(grad(psi_T), 1./(Re*Pr)*grad(T)))*dx


def residual(solution: sapphire.Solution, buoyancy: typing.Callable[[sapphire.Solution], typing.Any] = None):
    """Sum of the mass, momentum, and energy residuals"""
    if buoyancy is None:

        buoyancy = linear_boussinesq_buoyancy

    return mass_residual(solution) + momentum_residual(solution, buoyancy=buoyancy) + energy_residual(solution)


def dirichlet_boundary_conditions(solution: sapphire.Solution):
    """No-slip BCs for the velocity on every wall and constant temperature BCs on the left and right walls.

    The weak formulation does not admit Dirichlet boundary conditions on the pressure solution.
    To make the solution unique, the returned pressure solution will always be post-processed to have zero mean.
    """
    d = solution.mesh.geometric_dimension()

    return [
        fe.DirichletBC(solution.function_subspaces.u, (0,)*d, "on_boundary"),
        fe.DirichletBC(solution.function_subspaces.T, solution.ufl_constants.hotwall_temperature, MESH_HOTWALL_ID),
        fe.DirichletBC(solution.function_subspaces.T, solution.ufl_constants.coldwall_temperature, MESH_COLDWALL_ID)]


def solve_and_subtract_mean_pressure(problem: sapphire.Problem) -> sapphire.Solution:

    solution = sapphire.nonlinear_solve(problem)

    print("Subtracting mean pressure to make solution unique")

    p = solution.ufl_fields.p

    dx = fe.dx(degree=solution.quadrature_degree)

    mean_pressure = fe.assemble(p*dx)

    p = solution.subfunctions.p

    p = sapphire.helpers.assign_function_values(p - mean_pressure, p)

    print("Done subtracting mean pressure")

    return solution


def solve_with_rayleigh_number_continuation(problem: sapphire.Problem) -> sapphire.Solution:

    Ra = problem.solution.ufl_constants.rayleigh_number

    return sapphire.continuation.solve_with_bounded_continuation_sequence(
        problem=problem,
        nonlinear_solve=sapphire.nonlinear_solve,
        continuation_parameter_and_name=(Ra, 'Ra'),
        initial_sequence=(1, Ra.__float__()))


def solve(problem: sapphire.Problem) -> sapphire.Solution:

    return solve_with_rayleigh_number_continuation(problem)


def output(solution: sapphire.Solution, outdir_path: str = "sapphire_output/heat_driven_cavity/"):

    sapphire.plot(
        solution=solution,
        outdir_path=outdir_path)


def run_simulation(
        hotwall_temperature=0.5,
        coldwall_temperature=-0.5,
        reynolds_number=1.,
        rayleigh_number=1.e6,
        prandtl_number=0.71,
        mesh_dimensions=(20, 20),
        taylor_hood_pressure_element_degree=1,
        temperature_element_degree=2
        ) -> sapphire.Simulation:

    _mesh = mesh(mesh_dimensions)

    solution = sapphire.data.Solution(
        function=fe.Function(fe.FunctionSpace(
            _mesh,
            element(_mesh.ufl_cell(), taylor_hood_pressure_element_degree, temperature_element_degree))),
        function_component_names=SOLUTION_FUNCTION_COMPONENT_NAMES,
        ufl_constants=ufl_constants(
            hotwall_temperature=hotwall_temperature,
            coldwall_temperature=coldwall_temperature,
            reynolds_number=reynolds_number,
            rayleigh_number=rayleigh_number,
            prandtl_number=prandtl_number))

    problem = sapphire.data.Problem(
        solution=solution,
        residual=residual,
        dirichlet_boundary_conditions=dirichlet_boundary_conditions(solution),
        nullspace=nullspace(solution))

    sim = sapphire.data.Simulation(problem=problem, solutions=(solution,))

    return sapphire.run(sim=sim, solve=solve, output=output)


def verify_default_simulation():

    sim = run_simulation()

    solution = sim.solutions[0]

    Ra = solution.ufl_constants.rayleigh_number.__float__()

    Pr = solution.ufl_constants.prandtl_number.__float__()

    sapphire.helpers.verify_function_values_at_coordinates(
        function=sim.solutions[0].subfunctions.u,
        coordinates=[(0.5, y) for y in (0., 0.15, 0.34999, 0.5, 0.65, 0.84999)],
        # Checking y coordinates 0.3499 and 0.8499 instead of 0.35, 0.85 because the `firedrake.Function` evaluation fails at the exact coordinates. See https://github.com/firedrakeproject/firedrake/issues/1340
        expected_values=[(u_x*Ra**0.5/Pr, None) for u_x in (0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649)],
        absolute_tolerances=[(tol*Ra**0.5/Pr, None) for tol in (1.e-12, 0.001, 0.001, 1.e-12, 0.001, 0.001)])


if __name__ == '__main__':

    verify_default_simulation()
