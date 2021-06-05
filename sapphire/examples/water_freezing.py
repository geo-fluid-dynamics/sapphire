"""Water freezing example module"""
from typing import Tuple
from sapphire import Solution, Simulation, plot
from sapphire import solve as default_solve
from sapphire import find_working_continuation_parameter_value, solve_with_bounded_continuation_sequence, run
from sapphire.forms.natural_convection import element, COMPONENT_NAMES
from sapphire.forms.natural_convection import residual as natural_convection_residual
from sapphire.forms.enthalpy_porosity import residual as enthalpy_porosity_residual
from sapphire.examples.heat_driven_cavity import mesh, dirichlet_boundary_conditions, nullspace
from sapphire.examples.heat_driven_cavity import solve as solve_steady_heat_driven_cavity
from sapphire.examples.heat_driven_cavity_with_water import buoyancy_with_density_anomaly_of_water
from firedrake import ConvergenceError, Function


DEFAULT_SOLVER_PARAMETERS = {
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


def solve_with_auto_smoothing(sim: Simulation, initial_sequence: Tuple[float] = None) -> Tuple[Solution, Tuple[float]]:

    solution = sim.solutions[0]

    def search_upward():

        return find_working_continuation_parameter_value(
            sim=sim,
            solve=default_solve,
            continuation_parameter_and_name=(solution.ufl_constants.porosity_smoothing_factor, 'sigma'))

    def refine_downward(sequence: Tuple(float)):

        return solve_with_bounded_continuation_sequence(
            sim=sim,
            solve=default_solve,
            continuation_parameter_and_name=(solution.ufl_constants.porosity_smoothing_factor, 'sigma'),
            initial_sequence=sequence)

    initial_guess = Function(solution.function_space)

    initial_guess.assign(solution)

    sigma = solution.ufl_constants.porosity_smoothing_factor.__float__()

    if initial_sequence is None:
        # Find an over-regularization that works.
        working_value, solution = search_upward()

        if working_value == sigma:
            # No continuation was necessary.
            return solution

    # At this point, either a bounded sequence has been provided or a working upper bound has been found.
    # Next, continuation will be attempted using the sequence including refinement within the bounds.
    try:

        solution, sequence = refine_downward((working_value, sigma))

    except ConvergenceError as error:

        if initial_sequence is not None:
            # Try one more time without using the given sequence.
            # This is sometimes useful when trying to solve a later time step with a sequence that was only successful for an earlier time step.
            solution.assign(initial_guess)

            working_value, solution = search_upward()

            solution, sequence = refine_downward(sequence=(working_value, sigma))

        else:

            raise error

    # The above procedure was quite complicated.
    # Verify that the problem was solved with the correct regularization and that the simulation's attribute for this has been set to the correct value before returning.
    if solution.ufl_constants.porosity_smoothing_factor.__float__() != sigma:

        raise Exception("Continuation procedure ended on the wrong continuation parameter value.")

    return solution, sequence


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
        endtime=1.44,
        timestep_size=1.44/4.):

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
        residual=natural_convection_residual,
        dirichlet_boundary_conditions=dirichlet_boundary_conditions,
        nullspace=nullspace,
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
        dirichlet_boundary_conditions=dirichlet_boundary_conditions,
        nullspace=nullspace,
        quadrature_degree=4,
        initial_times=tuple(-i*timestep_size for i in range(time_discretization_stencil_size)),
        initial_values_functions=(initial_sim.solutions[0].function,)*time_discretization_stencil_size)

    output(sim.solutions[0])

    sim = run(sim=sim, solve=solve_with_auto_smoothing, endtime=endtime)

    return sim


def verify_default_simulation():

    sim = run_simulation()

    liquid_area = sim.solutions[0].post_processed_object['liquid_area']

    expected_liquid_area = 0.70

    print("Liquid area = {}".format(liquid_area))

    if round(liquid_area, 2) != expected_liquid_area:

        raise Exception("Expected liquid area {} but actual value is {}".format(expected_liquid_area, liquid_area))


if __name__ == '__main__':

    verify_default_simulation()
