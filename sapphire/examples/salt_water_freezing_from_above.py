from typing import Tuple
from sapphire import Mesh, Solution, Simulation, run, plot, report, write_checkpoint, solve_with_bounded_continuation_sequence, EutecticBinaryAlloy, MATERIALS
from sapphire import solve_with_timestep_size_continuation as _solve_with_timestep_size_continuation
from sapphire.examples.lid_driven_cavity import cavity_mesh, solve_and_subtract_mean_pressure, solve_with_lid_speed_continuation
from sapphire.examples.lid_driven_cavity import DEFAULT_FIREDRAKE_SOLVER_PARAMETERS as LID_DRIVEN_CAVITY_FIREDRAKE_SOLVER_PARAMETERS
from sapphire.forms.binary_alloy_enthalpy_porosity import INITIAL_SOLUTE_CONCENTRATION, COMPONENT_NAMES, element, postprocess, validate, thin_hele_shaw_cell_permeability, normalized_kozeny_carman_permeability
from sapphire.forms.binary_alloy_enthalpy_porosity import residual as default_residual
from firedrake import PeriodicRectangleMesh, MixedVectorSpaceBasis, VectorSpaceBasis, DirichletBC, cos, pi


SOLID_TO_LIQUID_HEAT_CAPACITY_RATIO = 1  # Assume equal heat capacity in solid and liquid

PARTITION_COEFFICIENT = 0.  # My formulation sets this to zero rather than an arbitrary small number.

PMWK_2019_FIXED_CHILL = {
    'frame_translation_velocity': (0., 0.),
    'solute_rayleigh_number': 1.e6,
    'temperature_rayleigh_number': 0.,
    'concentration_ratio': 2,
    'darcy_number': 1.e-4,
    'prandtl_number': 10,
    'stefan_number': 5,
    'lewis_number': 200,  # According to https://github.com/jrgparkinson/mushy-layer/blob/master/params/convergenceTest/FixedChill.parameters
    'reference_permeability': 1.e4,
    'thermal_conductivity_solid_to_liquid_ratio': 1.,
    'heat_capacity_solid_to_liquid_ratio': 1.,
    'partition_coefficient': 1.e-5,
    'initial_enthalpy': 6.3,  # This is above the initial liquidus enthalpy H_L_S0 = Ste - S_0 = 5 - (-1) = 6
    'top_wall_enthalpy': 2.5,  # Jamie said in an e-mail that this was deliberately set to the initial eutectic enthalpy, H_E = (1 + S_0/r)*Ste which indeed equals 2.5.
    'end_time': 0.0147,
    }


def liquidus_temperature(S, alloy: EutecticBinaryAlloy):

    T_e = alloy.eutectic_temperature

    T_m = alloy.melting_temperature_of_solvent

    S_e = alloy.eutectic_concentration

    m = (T_e - T_m)/S_e

    T_L = T_m + m*S

    return T_L


def nondimensional_temperature(S_0, T, alloy: EutecticBinaryAlloy):

    T_L_S0 = liquidus_temperature(S=S_0, alloy=alloy)

    T_e = alloy.eutectic_temperature

    return (T - T_e)/(T_L_S0 - T_e)


def nondimensional_enthalpy(Ste, T, f):

    c_sl = SOLID_TO_LIQUID_HEAT_CAPACITY_RATIO

    return Ste*f + (f + (1 - f)*c_sl)*T


def nondimensional_solute_concentration(S_0, S, alloy: EutecticBinaryAlloy):

    S_e = alloy.eutectic_concentration

    return (S - S_e)/(S_e - S_0)


DEFAULT_FIREDRAKE_SOLVER_PARAMETERS = {
    'snes_monitor': None,
    'snes_type': 'newtonls',
    'snes_linesearch_type': 'l2',
    'snes_linesearch_monitor': None,
    'snes_linesearch_maxstep': 1,
    'snes_linesearch_damping': 0.9,  # @todo Experiment with damping values (max 1)
    'snes_atol': 1.e-8,
    'snes_stol': 1.e-9,
    'snes_rtol': 1.e-7,
    'snes_max_it': 24,  # This should be higher for lower damping value (i.e. more damping)
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij'}


def output(solution: Solution):

    report(solution=solution, filepath_without_extension=solution.output_directory+'report')

    write_checkpoint(solution=solution, filepath_without_extension=solution.output_directory+'checkpoint')

    plot(solution=solution, output_directory_path=solution.output_directory+'plots/')


def periodic_mesh(nx, ny, Lx, Ly) -> Mesh:

    return Mesh(geometry=PeriodicRectangleMesh(nx, ny, Lx, Ly, direction="x", diagonal="crossed"), boundaries={'bottom': 1, 'top': 2})


def nullspace(solution: Solution):
    """Inform solver that pressure solution is not unique.

    It is only defined up to adding an arbitrary constant because there will be no boundary conditions on the pressure.
    """
    return MixedVectorSpaceBasis(
        solution.function.function_space(),
        [VectorSpaceBasis(constant=True), solution.function_subspaces.U, solution.function_subspaces.S, solution.function_subspaces.H])


def assign_initial_values(solution: Solution):

    solution.subfunctions.p.assign(0.)

    ihat, jhat = solution.unit_vectors

    solution.subfunctions.U.assign(0.*ihat + 0.*jhat)

    solution.subfunctions.S.assign(INITIAL_SOLUTE_CONCENTRATION)

    solution.subfunctions.H.assign(solution.ufl_constants.initial_enthalpy)


def residual_pmwk2019(solutions: Tuple[Solution]):

    return default_residual(solutions=solutions, permeability=thin_hele_shaw_cell_permeability)


def residual_sapphire2019(solutions: Tuple[Solution]):

    return default_residual(solutions=solutions, permeability=normalized_kozeny_carman_permeability)


def dirichlet_boundary_conditions_pmwk2019(solution: Solution):

    Gamma = solution.mesh.boundaries

    return (
        DirichletBC(solution.function_subspaces.U, (solution.ufl_constants.lid_speed, 0), Gamma['top']),
        DirichletBC(solution.function_subspaces.U, (0, 0), Gamma['bottom']),
        DirichletBC(solution.function_subspaces.H, solution.ufl_constants.top_wall_enthalpy, Gamma['top']),
        DirichletBC(solution.function_subspaces.H, solution.ufl_constants.initial_enthalpy, Gamma['bottom']))


def dirichlet_boundary_conditions_pmwk2019_modified(solution: Solution):
    """Mostly the BCs from PMWK2019 
    but with a spatial perturbation to the top wall enthalpy to make the periodic solution unique
    and with another BC on the top wall salinity to cover the case where it becomes decoupled from temperature (below eutectic enthalpy and above liquidus enthalpy).
    """
    Gamma = solution.mesh.boundaries

    x = solution.mesh.geometry.coordinates[0]

    Lx = solution.ufl_constants.periodic_cell_width

    H_top = solution.ufl_constants.top_wall_enthalpy

    pmag = solution.ufl_constants.top_wall_enthalpy_perturbation_relative_magnitude

    return (
        DirichletBC(solution.function_subspaces.U, (solution.ufl_constants.lid_speed, 0), Gamma['top']),
        DirichletBC(solution.function_subspaces.U, (0, 0), Gamma['bottom']),
        DirichletBC(solution.function_subspaces.H, H_top + pmag*H_top*cos(2*pi*x/Lx), Gamma['top']),
        DirichletBC(solution.function_subspaces.H, solution.ufl_constants.initial_enthalpy, Gamma['bottom']),
        DirichletBC(solution.function_subspaces.S, INITIAL_SOLUTE_CONCENTRATION, Gamma['top']))


def lid_driven_cavity_dirichlet_boundary_conditions(solution: Solution):

    Gamma = solution.mesh.boundaries

    return (
        DirichletBC(solution.function_subspaces.U, (solution.ufl_constants.lid_speed, 0), Gamma['top']),
        DirichletBC(solution.function_subspaces.U, (0, 0), (Gamma['bottom'], Gamma['left'], Gamma['right'])),
        )


def dirichlet_boundary_conditions_from_2019_sapphire_regression_test(solution: Solution):
    # The only difference is that I no longer constrain the pressure on the bottom wall, rather inform the solver of the nullspace and subtract mean pressure.
    # If I keep having trouble then I can try reverting to applying a Dirichlet BC to the pressure on the top wall instead of the nullspace.
    Gamma = solution.mesh.boundaries

    return (
        DirichletBC(solution.function_subspaces.U, (0, 0), (Gamma['top'], Gamma['bottom'])),
        DirichletBC(solution.function_subspaces.H, solution.ufl_constants.top_wall_enthalpy, Gamma['top']),
        DirichletBC(solution.function_subspaces.S, solution.ufl_constants.top_wall_solute_concentration, Gamma['top']),
        DirichletBC(solution.function_subspaces.H, solution.ufl_constants.initial_enthalpy, Gamma['bottom']),
        )


def lid_driven_flow_dirichlet_bcs(solution: Solution):

    Gamma = solution.mesh.boundaries

    return (
        DirichletBC(solution.function_subspaces.U, (1, 0), Gamma['top']),
        DirichletBC(solution.function_subspaces.U, (0, 0), Gamma['bottom']),
        )


def solve_with_top_wall_enthalpy_continuation(sim: Simulation):

    H_top = sim.solutions[0].ufl_constants.top_wall_enthalpy

    H_0 = sim.solutions[0].ufl_constants.initial_enthalpy

    solve_with_bounded_continuation_sequence(
        sim=sim,
        solve=solve_and_subtract_mean_pressure,
        continuation_parameter_and_name=(H_top, 'H_top'),
        initial_sequence=(H_0.__float__(), H_top.__float__()),
        start_index=0,
        output=output)


def solve_with_lewis_number_continuation(sim: Simulation):

    Le = sim.solutions[0].ufl_constants.lewis_number

    solve_with_bounded_continuation_sequence(
        sim=sim,
        solve=solve_and_subtract_mean_pressure,
        continuation_parameter_and_name=(Le, 'Le'),
        initial_sequence=(1, Le.__float__()),
        start_index=0)


def solve_with_solute_rayleigh_number_continuation(sim: Simulation):

    Ra_S = sim.solutions[0].ufl_constants.solute_rayleigh_number

    solve_with_bounded_continuation_sequence(
        sim=sim,
        solve=solve_and_subtract_mean_pressure,
        continuation_parameter_and_name=(Ra_S, 'Ra_S'),
        initial_sequence=(0., Ra_S.__float__()),
        maxcount=64,
        start_index=0)


def solve_with_porosity_smoothing_continuation(sim: Simulation):

    sigma = sim.solutions[0].ufl_constants.porosity_smoothing_factor

    solve_with_bounded_continuation_sequence(
        sim=sim,
        solve=solve_and_subtract_mean_pressure,
        continuation_parameter_and_name=(sigma, 'sigma'),
        initial_sequence=(4., 2., 1., sigma.__float__()),
        maxcount=24,
        start_index=0,
    )


def solve_with_timestep_size_continuation(sim: Simulation):

    _solve_with_timestep_size_continuation(sim=sim, solve=solve_and_subtract_mean_pressure, maxcount=8)


def solve_with_top_wall_enthalpy_and_timestep_size_continuation(sim: Simulation):

    H_top = sim.solutions[0].ufl_constants.top_wall_enthalpy

    H_0 = sim.solutions[0].ufl_constants.initial_enthalpy

    solve_with_bounded_continuation_sequence(
        sim=sim,
        solve=solve_with_timestep_size_continuation,
        continuation_parameter_and_name=(H_top, 'H_top'),
        initial_sequence=(H_0.__float__(), H_top.__float__()),
        start_index=0)


def solve_with_top_wall_enthalpy_and_solute_rayleigh_number_continuation(sim: Simulation):

    H_top = sim.solutions[0].ufl_constants.top_wall_enthalpy

    H_0 = sim.solutions[0].ufl_constants.initial_enthalpy

    solve_with_bounded_continuation_sequence(
        sim=sim,
        solve=solve_with_solute_rayleigh_number_continuation,
        continuation_parameter_and_name=(H_top, 'H_top'),
        initial_sequence=(H_0.__float__(), H_top.__float__()),
        start_index=0)


def solve_with_top_wall_enthalpy_and_porosity_smoothing_continuation(sim: Simulation):

    H_top = sim.solutions[0].ufl_constants.top_wall_enthalpy

    H_0 = sim.solutions[0].ufl_constants.initial_enthalpy

    solve_with_bounded_continuation_sequence(
        sim=sim,
        solve=solve_with_porosity_smoothing_continuation,
        continuation_parameter_and_name=(H_top, 'H_top'),
        initial_sequence=(H_0.__float__(), H_top.__float__()),
        start_index=0)


def solve_with_solute_rayleigh_number_and_porosity_smoothing_continuation(sim: Simulation):

    Ra_S = sim.solutions[0].ufl_constants.solute_rayleigh_number

    solve_with_bounded_continuation_sequence(
        sim=sim,
        solve=solve_with_porosity_smoothing_continuation,
        continuation_parameter_and_name=(Ra_S, 'Ra_S'),
        initial_sequence=(0., Ra_S.__float__()),
        maxcount=64,
        start_index=0)


def run_simulation(
        reference_permeability=PMWK_2019_FIXED_CHILL['reference_permeability'],
        thermal_conductivity_solid_to_liquid_ratio=PMWK_2019_FIXED_CHILL['thermal_conductivity_solid_to_liquid_ratio'],
        heat_capacity_solid_to_liquid_ratio=PMWK_2019_FIXED_CHILL['heat_capacity_solid_to_liquid_ratio'],
        prandtl_number=PMWK_2019_FIXED_CHILL['prandtl_number'],
        darcy_number=PMWK_2019_FIXED_CHILL['darcy_number'],
        lewis_number=PMWK_2019_FIXED_CHILL['lewis_number'],
        frame_translation_velocity=PMWK_2019_FIXED_CHILL['frame_translation_velocity'],
        stefan_number=PMWK_2019_FIXED_CHILL['stefan_number'],
        solute_rayleigh_number=PMWK_2019_FIXED_CHILL['solute_rayleigh_number'],
        temperature_rayleigh_number=PMWK_2019_FIXED_CHILL['temperature_rayleigh_number'],
        concentration_ratio=PMWK_2019_FIXED_CHILL['concentration_ratio'],
        initial_enthalpy=PMWK_2019_FIXED_CHILL['initial_enthalpy'],
        top_wall_enthalpy=PMWK_2019_FIXED_CHILL['top_wall_enthalpy'],
        top_wall_enthalpy_perturbation_relative_magnitude=0.01,
        mesh_width=0.2,
        mesh_height=0.4,
        endtime=PMWK_2019_FIXED_CHILL['end_time'],
        timestep_size=0.0001,
        time_discretization_stencil_size=3,
        nx=32,
        ny=64,
        taylor_hood_velocity_element_degree=2,
        enthalpy_element_degree=2,
        solute_element_degree=2,
        porosity_smoothing_factor=0.2,  # For the problem from pmwk2019, a value of 0.2 looks like it yields a good approximation; but it will be good to run a sensitivity study here.
        quadrature_degree=4,
        firedrake_solver_parameters=None,
        residual=residual_pmwk2019,
        mesh=periodic_mesh,
        dirichlet_boundary_conditions=dirichlet_boundary_conditions_pmwk2019_modified,
        solve_first_timestep=solve_with_top_wall_enthalpy_continuation,
        solve_during_run=solve_with_timestep_size_continuation,
        outdir='sapphire_output/salt_water_freezing_from_above/pmwk2019/',
        # The following were used for debugging.
        lid_speed=0.,
        top_wall_solute_concentration=None,
        ):

    if firedrake_solver_parameters is None:

        firedrake_solver_parameters = DEFAULT_FIREDRAKE_SOLVER_PARAMETERS

    _mesh = mesh(nx=nx, ny=ny, Lx=mesh_width, Ly=mesh_height)

    _element = element(cell=_mesh.cell, taylor_hood_velocity_degree=taylor_hood_velocity_element_degree, solute_degree=solute_element_degree, enthalpy_degree=enthalpy_element_degree)

    ufl_constants = {
        'partition_coefficient': PARTITION_COEFFICIENT,
        'reference_permeability': reference_permeability,
        'concentration_ratio': concentration_ratio,
        'initial_enthalpy': initial_enthalpy,
        'top_wall_enthalpy': top_wall_enthalpy,
        'top_wall_enthalpy_perturbation_relative_magnitude': top_wall_enthalpy_perturbation_relative_magnitude,
        'temperature_rayleigh_number': temperature_rayleigh_number,
        'solute_rayleigh_number': solute_rayleigh_number,
        'stefan_number': stefan_number,
        'prandtl_number': prandtl_number,
        'darcy_number': darcy_number,
        'lewis_number': lewis_number,
        'frame_translation_velocity': frame_translation_velocity,
        'heat_capacity_solid_to_liquid_ratio': heat_capacity_solid_to_liquid_ratio,
        'thermal_conductivity_solid_to_liquid_ratio': thermal_conductivity_solid_to_liquid_ratio,
        'periodic_cell_width': mesh_width,
        'porosity_smoothing_factor': porosity_smoothing_factor,
        'timestep_size': timestep_size,
        # The following were used for debugging
        'lid_speed': lid_speed,
        'top_wall_solute_concentration': top_wall_solute_concentration,
        }

    sim = Simulation(
        mesh=_mesh,
        element=_element,
        solution_component_names=COMPONENT_NAMES,
        ufl_constants=ufl_constants,
        residual=residual,
        quadrature_degree=quadrature_degree,
        dirichlet_boundary_conditions=dirichlet_boundary_conditions,
        nullspace=nullspace,
        firedrake_solver_parameters=firedrake_solver_parameters,
        time_discretization_stencil_size=time_discretization_stencil_size,
        output_directory=outdir,
        )

    for solution in sim.solutions:

        assign_initial_values(solution)

        solution.post_processed_objects = postprocess(solution)

        output(solution)

        validate(solution)

    run(sim=sim, endtime=timestep_size, solve=solve_first_timestep, postprocess=postprocess, validate=validate, output=output)

    run(sim=sim, endtime=endtime, solve=solve_during_run, postprocess=postprocess, validate=validate, output=output)

    return sim


def run_diffusive_solidification_simulation():

    return run_simulation(
        residual=residual_pmwk2019,
        dirichlet_boundary_conditions=dirichlet_boundary_conditions_pmwk2019,
        solute_rayleigh_number=0.,
        solve_first_timestep=solve_and_subtract_mean_pressure,
        solve_during_run=solve_and_subtract_mean_pressure,
        outdir='sapphire_output/salt_water_freezing_from_above/diffusive_solidification/')


def run_lid_driven_flow_simulation():

    endtime = 1.e12

    return run_simulation(
        residual=residual_sapphire2019,
        dirichlet_boundary_conditions=lid_driven_flow_dirichlet_bcs,
        solute_rayleigh_number=0.,
        temperature_rayleigh_number=0.,
        darcy_number=1.e12,
        endtime=endtime,
        timestep_size=endtime,
        solve_first_timestep=solve_and_subtract_mean_pressure,
        outdir='sapphire_output/salt_water_freezing_from_above/lid_driven_flow/',
    )


def run_lid_driven_cavity_simulation(lid_speed, meshsize):

    endtime = 1.e12

    Ste = PMWK_2019_FIXED_CHILL['stefan_number']

    H_L = Ste - INITIAL_SOLUTE_CONCENTRATION

    # Oddly, we can't I set the enthalpy at exactly H_L or far above H_L. I'm going to throw that into the backlog.
    # H = H_L + 3.  # This doesn't work
    # H = H_L + 2.  # This doesn't work
    H = H_L + 1.5  # This works
    # H = H_L + 1.  # This works
    # H = H_L + 0.1  # This works
    # H = H_L + 0.01  # This works
    # H = H_L  # This doesn't work.
    # H = 0.99999*H_L  # This works
    # H = 0.9999*H_L  # This works
    # H = 0.999*H_L  # This works
    # H = 0.99*H_L  # This works
    # H = 0.9*H_L  # This works

    return run_simulation(
        residual=residual_sapphire2019,
        mesh=cavity_mesh,
        mesh_width=1,
        mesh_height=1,
        nx=meshsize,
        ny=meshsize,
        dirichlet_boundary_conditions=lid_driven_cavity_dirichlet_boundary_conditions,
        initial_enthalpy=H,
        top_wall_enthalpy=H,
        lid_speed=lid_speed,
        stefan_number=Ste,
        prandtl_number=2.,  # Momentum diffusion term has coefficient Pr in binary alloy enthalpy porosity form, 2/Re in incompressible flow form
        # concentration_ratio=10.,
        solute_rayleigh_number=0.,
        temperature_rayleigh_number=0.,
        darcy_number=1.e16,
        lewis_number=0.001,
        porosity_smoothing_factor=0.2,  # This works
        # porosity_smoothing_factor=0.1,  # This works
        # porosity_smoothing_factor=0.01,  # This works
        # porosity_smoothing_factor=0.001,  # This works
        endtime=endtime,
        timestep_size=endtime,
        solve_first_timestep=solve_with_lid_speed_continuation,
        firedrake_solver_parameters=LID_DRIVEN_CAVITY_FIREDRAKE_SOLVER_PARAMETERS,
        outdir='sapphire_output/salt_water_freezing_from_above/lid_driven_cavity_u{}_n{}/'.format(lid_speed, meshsize),
    )


def run_draft2019_regression_simulation(
        solve_first_timestep,
        solve_during_run,
        solute_rayleigh_number=5.e6,
        temperature_rayleigh_number=1.e6,
        outdir='sapphire_output/salt_water_freezing_from_above/draft2019/'
        ):

    T_L = liquidus_temperature

    S_0_dim = 3.8  # [% wt. NaCl]

    alloy = MATERIALS.sodium_chloride_dissolved_in_water

    T_m = nondimensional_temperature(S_0=S_0_dim, T=T_L(S=S_0_dim, alloy=alloy), alloy=alloy)

    T_0 = T_m

    Ste = 0.27

    phi_0 = 1.

    H_0 = nondimensional_enthalpy(Ste, T_0, phi_0)

    print("H_0 = {}".format(H_0))

    # @todo How did pmwk2019 properly constraint the top wall solution with only prescribing enthalpy? Also either the concentration or the porosity must be prescribed for the solution to be unique.
    # I e-mailed Jamie asking about it.
    # He replied clarifying that the enthalpy BC they used (and reported) was the eutectic enthalpy

    def eutectic_porosity(S, r):

        return 1 + S/r

    def eutectic_enthalpy(phi_e, Ste):

        return phi_e * Ste

    r = 1.2

    phi_top = 0.001

    S_top = r*(phi_top - 1.)  # phi_top=0.001 was used to compute S_l_top as reported in my 2019 thesis draft and as shown in the regression test code.
    # Sice I constrain the enthalpy to the eutectic at the top wall, I can compute S_top from phi_top which is equivalent to phi_e.

    print("S_top = {}".format(S_top))

    H_top = eutectic_enthalpy(eutectic_porosity(S_top, r), Ste)

    print("H_top = {}".format(H_top))

    return run_simulation(
        reference_permeability=1.e4,  # This isn't used in Le Bars's model or in the model from my 2019 draft, but it is used by pmwk2019
        thermal_conductivity_solid_to_liquid_ratio=1,
        heat_capacity_solid_to_liquid_ratio=1,
        prandtl_number=7,
        darcy_number=1.e-4,
        lewis_number=80,
        frame_translation_velocity=(0., 0.),
        stefan_number=Ste,
        solute_rayleigh_number=solute_rayleigh_number,
        temperature_rayleigh_number=temperature_rayleigh_number,
        concentration_ratio=r,
        initial_enthalpy=H_0,
        top_wall_enthalpy=H_top,
        top_wall_solute_concentration=S_top,
        mesh_width=0.1,
        mesh_height=0.2,
        taylor_hood_velocity_element_degree=2,
        enthalpy_element_degree=1,
        solute_element_degree=1,
        nx=10,
        ny=20,
        time_discretization_stencil_size=2,
        timestep_size=0.001,
        endtime=0.025,
        porosity_smoothing_factor=0.1,  # @todo Try setting a smaller value. Might need to use an inner continuation loop.
        quadrature_degree=4,
        firedrake_solver_parameters=None,
        solve_first_timestep=solve_first_timestep,
        solve_during_run=solve_during_run,
        outdir=outdir,
        )


if __name__ == '__main__':

    # plot_phase_diagram_with_and_without_smoothing(
    #     heat_capacity_solid_to_liquid_ratio=1.,
    #     concentration_ratio=2.,
    #     stefan_number=3.,
    #     porosity_smoothing_factor=0.5,
    #     output_directory_path='sapphire_output/salt_water_freezing_from_above/phase_diagram/')

    # Lid driven cavity using the BAS equations looks right.
    # run_lid_driven_cavity_simulation(lid_speed=1000, meshsize=50)

    # This fails, keeps getting stuck at H_top ~= 6
    # run_simulation()

    # Try  lower order bases because that has been more robust previously
    run_simulation(endtime=0.015, timestep_size=0.001, nx=20, ny=40, solute_element_degree=1, enthalpy_element_degree=1, time_discretization_stencil_size=2)

    # next: try more refinement
    # run_simulation(nx=64, ny=128, solute_element_degree=1, enthalpy_element_degree=1, time_discretization_stencil_size=2)

    # Try to see if top wall enthalpy continuation always gets stuck near the liquidus enthalpy (at initial concentration)
    # Ste = 3.

    # r = PMWK_2019_FIXED_CHILL['concentration_ratio']

    # S_0 = INITIAL_SOLUTE_CONCENTRATION

    # phi_E_S0 = 1 + S_0/r

    # H_E_S0 = phi_E_S0*Ste

    # H_L_S0 = Ste - S_0

    # print("Ste = {}, H_E_S0 = {}, H_L_S0 = {}".format(Ste, H_E_S0, H_L_S0))

    # run_simulation(stefan_number=Ste, concentration_ratio=r, initial_enthalpy=H_L_S0 + 0.3, top_wall_enthalpy=H_E_S0)
    # This got stuck around H_top = 2.89. Here, H_L_S0 = 4, so the continuation issue seems unrelated to the liquidus enthalpy (which is good!)

    # Then: Try diffusive solidification benchmark from Parkinson's paper/thesis.
    # Then: Try porous media flow benchmark from Parkinson's paper/thesis.

    # Old Notes:

    # Diffusive solidification seems to work fine without any continuation (though I didn't try very small sigma)!
    # run_diffusive_solidification_simulation()

    # Maybe it's a better idea to reproduce my own brine plume simulation.
    # run_draft2019_regression_simulation(
    #     solve_first_timestep=solve_with_timestep_size_continuation,
    #     solve_during_run=solve_with_timestep_size_continuation)

    # run_draft2019_regression_simulation(
    #     solve_first_timestep=solve_with_top_wall_enthalpy_continuation,
    #     solve_during_run=solve_with_timestep_size_continuation)

    # This worked well! The porosity smoothing continuation was necessary.
    # run_draft2019_regression_simulation(
    #     solve_first_timestep=solve_with_porosity_smoothing_continuation,
    #     solve_during_run=solve_with_porosity_smoothing_continuation,
    #     solute_rayleigh_number=0.,
    #     temperature_rayleigh_number=0.,
    #     outdir='sapphire_output/diffusive_solidification/draft2019/',
    #     )

    # Nothing is working. Something has to be wrong which isn't part of the diffusive solidification problem.
    # run_draft2019_regression_simulation(
    #     solve_first_timestep=solve_with_solute_rayleigh_number_and_porosity_smoothing_continuation,
    #     solve_during_run=solve_with_solute_rayleigh_number_and_porosity_smoothing_continuation,
    #     temperature_rayleigh_number=0.,
    #     )

    # None of the pmwk2019 simulations are working. I can't solve the first timestep. I've tried many combinations of continuation procedures.
    # run_simulation(
    #     residual=residual_pmwk2019,
    #     dirichlet_boundary_conditions=dirichlet_boundary_conditions_almost_pmwk2019_but_fixed_top_salinity)

    # run_simulation(
    #     residual=residual_pmwk2019,
    #     dirichlet_boundary_conditions=dirichlet_boundary_conditions_almost_pmwk2019_but_fixed_top_salinity,
    #     ny=128)

    # run_simulation(
    #    residual=residual_pmwk2019,
    #    dirichlet_boundary_conditions=dirichlet_boundary_conditions_almost_pmwk2019_but_fixed_top_salinity,
    #    solve_first_timestep=solve_with_rayleigh_number_continuation,
    #    solve_during_run=solve_with_rayleigh_number_continuation,
    #    nx=100,
    #    ny=200,
    #    taylor_hood_velocity_element_degree=2,
    #    enthalpy_element_degree=1,
    #    solute_element_degree=1,
    #    time_discretization_stencil_size=2,
    #    timestep_size=0.001,
    #    endtime=0.015,
    # )

    # run_simulation(
    #     residual=residual_pmwk2019,
    #     dirichlet_boundary_conditions=dirichlet_boundary_conditions_almost_pmwk2019_but_fixed_top_salinity,
    #     solve_first_timestep=solve_with_top_wall_enthalpy_and_timestep_size_continuation,
    #     solve_during_run=solve_with_timestep_size_continuation)
