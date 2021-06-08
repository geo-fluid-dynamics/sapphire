"""Water freezing example module"""
from sapphire import Solution, Simulation, plot, run, find_working_continuation_parameter_value, solve_with_bounded_continuation_sequence, ContinuationError
from sapphire import solve as default_solve
from sapphire.forms.natural_convection import element, COMPONENT_NAMES
from sapphire.forms.enthalpy_porosity import postprocess
from sapphire.forms.enthalpy_porosity import residual as enthalpy_porosity_residual
from sapphire.examples.heat_driven_cavity import mesh, dirichlet_boundary_conditions, nullspace
from sapphire.examples.heat_driven_cavity import solve as solve_steady_heat_driven_cavity
from sapphire.examples.heat_driven_cavity_with_water import buoyancy_with_density_anomaly_of_water
from sapphire.examples.heat_driven_cavity_with_water import residual as steady_state_heat_driven_cavity_residual
from sapphire.examples.heat_driven_cavity_with_water import DEFAULT_FIREDRAKE_SOLVER_PARAMTERS as DEFAULT_HEAT_DRIVEN_CAVITY_FIREDRAKE_SOLVER_PARAMETERS
from firedrake import Function


DEFAULT_WATER_FREEZING_FIREDRAKE_SOLVER_PARAMETERS = {
    'snes_monitor': None,
    'snes_type': 'newtonls',
    'snes_linesearch_type': 'l2',
    'snes_linesearch_maxstep': 1,
    'snes_linesearch_damping': 1,
    'snes_atol': 1.e-9,
    'snes_stol': 1.e-9,
    'snes_rtol': 0.,
    'snes_max_it': 24,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij'}


LAST_SUCCESSFUL_SMOOTHING_SEQUENCE = None


def solve_with_auto_smoothing(sim: Simulation) -> Solution:

    solution = sim.solutions[0]

    initial_sequence = solution.extras['sigma_continuation_sequence']

    def search_upward():

        return find_working_continuation_parameter_value(
            sim=sim,
            solve=default_solve,
            continuation_parameter_and_name=(solution.ufl_constants.porosity_smoothing_factor, 'sigma'))

    def refine_downward(start_index):

        return solve_with_bounded_continuation_sequence(
            sim=sim,
            solve=default_solve,
            continuation_parameter_and_name=(solution.ufl_constants.porosity_smoothing_factor, 'sigma'),
            start_index=start_index,
            initial_sequence=solution.extras['sigma_continuation_sequence'])

    initial_guess = Function(solution.function_space)

    initial_guess.assign(solution.function)

    sigma = solution.ufl_constants.porosity_smoothing_factor.__float__()

    if initial_sequence is None:

        working_value = search_upward()

        if working_value == sigma:
            # No continuation was necessary.
            return solution

        solution.ufl_constants.porosity_smoothing_factor.assign(sigma)

        solution.extras['sigma_continuation_sequence'] = (working_value, sigma)

        solution, solution.extras['sigma_continuation_sequence'] = refine_downward(start_index=1)

        return solution

    try:

        solution, solution.extras['sigma_continuation_sequence'] = refine_downward(start_index=0)

    except ContinuationError as exception:

        if initial_sequence is not None:
            # Attempt searching for a working value and refining a new sequence rather than using the provided initial sequence.
            solution.function.assign(initial_guess)

            solution.ufl_constants.porosity_smoothing_factor.assign(sigma)

            working_value = search_upward()

            solution.ufl_constants.porosity_smoothing_factor.assign(sigma)

            solution.extras['sigma_continuation_sequence'] = (working_value, sigma)

            solution, solution.extras['sigma_continuation_sequence'] = refine_downward(start_index=1)

        else:

            raise Exception("Failed to find a working continuation sequence.") from exception

    # The above procedure was quite complicated.
    # Verify that the problem was solved with the correct regularization and that the simulation's attribute for this has been set to the correct value before returning.
    if solution.ufl_constants.porosity_smoothing_factor.__float__() != sigma:

        raise Exception("Continuation procedure ended on the wrong continuation parameter value.")

    return solution


def output(solution: Solution):

    plot(solution=solution, outdir_path="sapphire_output/water_freezing/plots/")


def residual(sim: Simulation):

    return enthalpy_porosity_residual(sim, buoyancy=buoyancy_with_density_anomaly_of_water)


def run_simulation(
        reynolds_number=1.,
        rayleigh_number=2.518084e6,
        prandtl_number=6.99,
        stefan_number=0.125,
        melting_temperature=0.,
        density_solid_to_liquid_ratio=916.70/999.84,
        heat_capacity_solid_to_liquid_ratio=0.500,
        thermal_conductivity_solid_to_liquid_ratio=2.14/0.561,
        reference_temperature_range__degC=10.,
        hotwall_temperature=1.,
        coldwall_temperature_before_freezing=0.,
        coldwall_temperature_during_freezing=-1.,
        solid_velocity_relaxation_factor=1.e-12,
        porosity_smoothing_factor=0.005,
        mesh_dimensions=(24, 24),
        taylor_hood_velocity_element_degree=2,
        temperature_element_degree=2,
        time_discretization_stencil_size=3,
        quadrature_degree=4,
        heat_driven_cavity_firedrake_solver_parameters=None,
        water_freezing_firedrake_solver_parameters=None,
        endtime=1.44,
        timestep_size=1.44/4.):

    if heat_driven_cavity_firedrake_solver_parameters is None:

        heat_driven_cavity_firedrake_solver_parameters = DEFAULT_HEAT_DRIVEN_CAVITY_FIREDRAKE_SOLVER_PARAMETERS

    if water_freezing_firedrake_solver_parameters is None:

        water_freezing_firedrake_solver_parameters = DEFAULT_WATER_FREEZING_FIREDRAKE_SOLVER_PARAMETERS

    _mesh = mesh(mesh_dimensions)

    _element = element(_mesh.cell, taylor_hood_velocity_element_degree, temperature_element_degree)

    initial_sim = Simulation(
        mesh=_mesh,
        element=_element,
        solution_component_names=COMPONENT_NAMES,
        ufl_constants={
            'reynolds_number': reynolds_number,
            'rayleigh_number': rayleigh_number,
            'prandtl_number': prandtl_number,
            'hotwall_temperature': hotwall_temperature,
            'coldwall_temperature': coldwall_temperature_before_freezing,
            'reference_temperature_range__degC': reference_temperature_range__degC},
        residual=steady_state_heat_driven_cavity_residual,
        quadrature_degree=quadrature_degree,
        dirichlet_boundary_conditions=dirichlet_boundary_conditions,
        nullspace=nullspace,
        firedrake_solver_parameters=heat_driven_cavity_firedrake_solver_parameters,
        initial_times=None)

    initial_sim = run(sim=initial_sim, solve=solve_steady_heat_driven_cavity)

    sim = Simulation(
        mesh=_mesh,
        element=_element,
        solution_component_names=COMPONENT_NAMES,
        ufl_constants={
            'reynolds_number': reynolds_number,
            'rayleigh_number': rayleigh_number,
            'prandtl_number': prandtl_number,
            'hotwall_temperature': hotwall_temperature,
            'coldwall_temperature': coldwall_temperature_during_freezing,
            'reference_temperature_range__degC': reference_temperature_range__degC,
            'stefan_number': stefan_number,
            'melting_temperature': melting_temperature,
            'density_solid_to_liquid_ratio': density_solid_to_liquid_ratio,
            'heat_capacity_solid_to_liquid_ratio': heat_capacity_solid_to_liquid_ratio,
            'thermal_conductivity_solid_to_liquid_ratio': thermal_conductivity_solid_to_liquid_ratio,
            'solid_velocity_relaxation_factor': solid_velocity_relaxation_factor,
            'porosity_smoothing_factor': porosity_smoothing_factor},
        residual=residual,
        quadrature_degree=quadrature_degree,
        dirichlet_boundary_conditions=dirichlet_boundary_conditions,
        nullspace=nullspace,
        firedrake_solver_parameters=water_freezing_firedrake_solver_parameters,
        initial_times=tuple((1 - i)*timestep_size for i in range(time_discretization_stencil_size)),
        initial_values_functions=(initial_sim.solutions[0].function,)*time_discretization_stencil_size)

    sim.solutions[0].post_processed_objects = postprocess(sim.solutions[0])

    output(sim.solutions[0])

    sim.solutions[0].extras['sigma_continuation_sequence'] = None

    sim = run(sim=sim, endtime=endtime, solve=solve_with_auto_smoothing, postprocess=postprocess, output=output)

    return sim


def verify_default_simulation():

    sim = run_simulation()

    liquid_area = sim.solutions[0].post_processed_objects['liquid_area']

    expected_liquid_area = 0.70

    print("Liquid area = {}".format(liquid_area))

    if round(liquid_area, 2) != expected_liquid_area:

        raise Exception("Expected liquid area {} but actual value is {}".format(expected_liquid_area, liquid_area))


if __name__ == '__main__':

    verify_default_simulation()
