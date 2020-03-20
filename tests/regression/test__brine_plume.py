""" A regression test which simulates a brine plume """
import firedrake as fe
import sapphire.simulations.convection_coupled_alloy_phasechange
import sapphire.benchmarks.freeze_salt_water_from_above


tempdir = sapphire.test.datadir

equations_module = sapphire.simulations.convection_coupled_alloy_phasechange

benchmark_module = sapphire.benchmarks.freeze_salt_water_from_above

def dirichlet_boundary_conditions(sim):

    W = sim.function_space
    
    h_h = equations_module.enthalpy(
        sim = sim,
        temperature = sim.max_temperature,
        porosity = benchmark_module.initial_porosity)
    
    f_lc = sim.minimum_porosity
    
    T_c = sim.cold_wall_temperature
    
    h_c = equations_module.enthalpy(
        sim = sim,
        temperature = T_c,
        porosity = f_lc)
    
    S_lc = equations_module.mushy_layer_liquid_solute_concentration(
        sim, temperature = T_c)
    
    return [
        fe.DirichletBC(W.sub(1), (0., 0.), 2),
        fe.DirichletBC(W.sub(2), h_c, 2),
        fe.DirichletBC(W.sub(3), S_lc, 2),
        fe.DirichletBC(W.sub(0), 0., 1),
        fe.DirichletBC(W.sub(1), (0., 0.), 1),
        fe.DirichletBC(W.sub(2), h_h, 1)]
        

T_m = 0.  # [deg C]
    
T_e = -21.  # [deg C]

S_e = 23.  # [% wt. NaCl]

m = (T_e - T_m)/S_e
    
def T_L(S):
    
    return T_m + m*S

def S__nondim(S):

    return S/S_e
    
k_l = 0.54  # [W/(m K)]

c_l = 4200.  # [J/(kg K)]

rho = 1000.  # [kg/m^3]

c_sl = 1.

h_m = 3.3e5  # [J/kg]

Le = 80.

Pr = 7.

T_h = T_m  # [deg C]

Ste = c_l*(T_h - T_e)/h_m

def test__brine_plume(tempdir):
    
    S_0 = 3.8  # [% wt. NaCl]
    
    def T__nondim(T):
    
        return (T - T_e)/(T_h - T_e)
    
    Delta_t = 0.001
    
    solver_parameters = sapphire.simulations.\
        convection_coupled_alloy_phasechange.default_solver_parameters
    
    solver_parameters["snes_view"] = None

    solver_parameters["snes_linesearch_monitor"] = None
    
    sim = benchmark_module.Simulation(
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        darcy_number = 1.e-4,
        lewis_number = Le,
        prandtl_number = Pr,
        thermal_rayleigh_number = 1.e6,
        solutal_rayleigh_number = 5.e6,
        stefan_number = Ste,
        pure_liquidus_temperature = T__nondim(T_m),
        thermal_conductivity_solid_to_liquid_ratio = 1.,
        initial_solute_concentration = S__nondim(S_0),
        cold_wall_temperature = T__nondim(T_e + 1.),
        cold_wall_porosity = 0.001,
        porosity_smoothing = 0.001,
        timestep_size = Delta_t,
        Lx = 0.1,
        Ly = 0.2,
        nx = 10, 
        ny = 20,
        mesh_diagonal = "crossed",
        pressure_penalty_factor = 0.,
        element_degrees = (1, 2, 1, 1), 
        quadrature_degree = 4,
        adaptive_timestep_minimum = 1.e-6,
        solver_parameters = solver_parameters,
        output_directory_path = tempdir)
    
    sim.run(endtime = 0.025, solve_with_adaptive_timestep = True)
    
    tolerance = 0.5
    
    print("Max speed = {}".format(sim.max_speed))
    
    assert(abs(sim.max_speed - 39.) < tolerance)
    