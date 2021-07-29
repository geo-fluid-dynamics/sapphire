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
from sapphire import Mesh, Solution, Simulation, solve_with_bounded_continuation_sequence, run, plot
from sapphire import solve as default_solve
from sapphire.forms.incompressible_flow import COMPONENT_NAMES, element, residual
from firedrake import RectangleMesh, DirichletBC, MixedVectorSpaceBasis, VectorSpaceBasis, dx, assemble


DEFAULT_FIREDRAKE_SOLVER_PARAMETERS = {
    'snes_type': 'newtonls',
    'snes_max_it': 16,
    'snes_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'mat_type': 'aij',
    'pc_factor_mat_solver_type': 'mumps'}


def cavity_mesh(nx: int, ny: int, Lx: float, Ly: float):

    return Mesh(geometry=RectangleMesh(nx, ny, Lx, Ly, diagonal="crossed"), boundaries={'left': 1, 'right': 2, 'bottom': 3, 'top': 4})


def dirichlet_boundary_conditions(solution: Solution):
    """No-slip BCs for the velocity on every wall and constant temperature BCs on the left and right walls.

    The weak formulation does not admit Dirichlet boundary conditions on the pressure solution.
    To make the solution unique, the returned pressure solution will always be post-processed to have zero mean.
    """
    Gamma = solution.mesh.boundaries

    return (
        DirichletBC(solution.function_subspaces.U, (solution.ufl_constants.lid_speed, 0), Gamma['top']),
        DirichletBC(solution.function_subspaces.U, (0, 0), (Gamma['bottom'], Gamma['left'], Gamma['right']))
        )


def nullspace(solution: Solution):
    """Inform solver that pressure solution is not unique.

    It is only defined up to adding an arbitrary constant because there will be no boundary conditions on the pressure.
    """
    return MixedVectorSpaceBasis(
        solution.function.function_space(),
        [VectorSpaceBasis(constant=True), solution.function_subspaces.U])


def solve_and_subtract_mean_pressure(sim: Simulation):

    default_solve(sim)

    solution = sim.solutions[0]

    print("Subtracting mean pressure to make solution unique")

    p = solution.ufl_fields.p

    mean_pressure = assemble(p*dx(degree=solution.quadrature_degree))

    p = solution.subfunctions.p

    p.assign(p - mean_pressure)

    print("Done subtracting mean pressure")


def solve_with_reynolds_number_continuation(sim: Simulation):

    Re = sim.solutions[0].ufl_constants.reynolds_number

    solve_with_bounded_continuation_sequence(
        sim=sim,
        solve=default_solve,
        continuation_parameter_and_name=(Re, 'Re'),
        initial_sequence=(1., Re.__float__()))


def solve_with_lid_speed_continuation(sim: Simulation):

    lid_speed = sim.solutions[0].ufl_constants.lid_speed

    solve_with_bounded_continuation_sequence(
        sim=sim,
        solve=solve_and_subtract_mean_pressure,
        continuation_parameter_and_name=(lid_speed, 'lid_speed'),
        initial_sequence=(0., lid_speed.__float__()),
        start_index=0)


def output(solution: Solution):

    plot(solution=solution, output_directory_path=solution.output_directory+'/plots/')


def run_simulation(
        reynolds_number,
        lid_speed,
        meshsize,
        solve=solve_with_reynolds_number_continuation,
        taylor_hood_velocity_element_degree=2,
        outdir='sapphire_output/lid_driven_cavity/'):

    outdir += '/Re{}_u{}_n{}/'.format(reynolds_number, lid_speed, meshsize)

    _mesh = cavity_mesh(nx=meshsize, ny=meshsize, Lx=1, Ly=1)

    sim = Simulation(
        mesh=_mesh,
        element=element(_mesh.cell, taylor_hood_velocity_element_degree),
        solution_component_names=COMPONENT_NAMES,
        ufl_constants={
            'reynolds_number': reynolds_number,
            'lid_speed': lid_speed,
            },
        residual=residual,
        dirichlet_boundary_conditions=dirichlet_boundary_conditions,
        nullspace=nullspace,
        firedrake_solver_parameters=DEFAULT_FIREDRAKE_SOLVER_PARAMETERS,
        time_discretization_stencil_size=1,
        output_directory=outdir)

    return run(sim=sim, solve=solve, output=output)


if __name__ == '__main__':

    # run_simulation(meshsize=50, reynolds_number=1000, lid_speed=1)
    run_simulation(meshsize=50, reynolds_number=1, lid_speed=100, solve=solve_with_lid_speed_continuation)
