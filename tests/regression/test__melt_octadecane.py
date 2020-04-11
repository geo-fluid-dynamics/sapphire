import sapphire.simulations.examples.melt_octadecane
import sapphire.test


tempdir = sapphire.test.datadir

def test__validate__melt_octadecane__regression(tempdir):
    
    sim = sapphire.simulations.examples.melt_octadecane.Simulation(
        element_degree = (1, 1, 1),
        meshsize = 24,
        quadrature_degree = 4,
        time_stencil_size = 3,
        timestep_size = 20.,
        liquidus_smoothing_factor = 1./200.,
        solid_velocity_relaxation_factor = 1.e-12,
        output_directory_path = tempdir)
    
    sim.run(endtime = 80.)
    
    print("Liquid area = {}".format(sim.liquid_area))
    
    assert(abs(sim.liquid_area - 0.41) < 0.01)
  