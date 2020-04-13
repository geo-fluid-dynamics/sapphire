import firedrake as fe 
import sapphire.simulations.examples.freeze_water_in_cavity
import sapphire.test


tempdir = sapphire.test.datadir

def test__freeze_water__regression(tempdir):
    
    endtime = 1.44
    
    sim = sapphire.simulations.examples.freeze_water_in_cavity.Simulation(
        element_degree = (1, 2, 2),
        mesh = fe.UnitSquareMesh(24, 24),
        quadrature_degree = 4,
        time_stencil_size = 3,
        timestep_size = endtime/4.,
        solid_velocity_relaxation_factor = 1.e-12,
        liquidus_smoothing_factor = 1./200.,
        output_directory_path = tempdir)
    
    sim.run(endtime = endtime, plot = True)
    
    print("Liquid area = {}".format(sim.liquid_area))
    
    assert(abs(sim.liquid_area - 0.69) < 0.01)
    
    
def test__freeze_water_3d__regression(tempdir):
    
    X = 0.038  # m

    nu_l = 1.0032e-6  # m^2/s

    timescale = X**2/nu_l
    
    target_time = 2340.  # s
    
    nondimensional_target_time = target_time/timescale
    
    sim = sapphire.simulations.examples.freeze_water_in_cavity.Simulation(
        element_degree = (1, 1, 1),
        mesh = fe.UnitCubeMesh(8, 8, 8),
        quadrature_degree = 2,
        time_stencil_size = 3,
        timestep_size = nondimensional_target_time/2.,
        solid_velocity_relaxation_factor = 1.e-8,
        pressure_penalty_factor = 1.e-7,
        liquidus_smoothing_factor = 0.01,
        output_directory_path = tempdir)
    
    sim.run(endtime = nondimensional_target_time)
    
    print("Liquid area = {}".format(sim.liquid_area))
    
    assert(abs(sim.liquid_area - 0.67) < 0.01)
    