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
        liquidus_smoothing_factor = 0.005,
        output_directory_path = tempdir)
    
    sim.run(endtime = endtime, write_plots = True)
    
    print("Liquid area = {}".format(sim.liquid_area))
    
    assert(abs(sim.liquid_area - 0.69) < 0.01)
