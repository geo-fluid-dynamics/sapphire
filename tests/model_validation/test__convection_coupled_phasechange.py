import firedrake as fe 
import sapphire.mms
import sapphire.benchmarks.melt_octadecane_in_cavity
import sapphire.benchmarks.freeze_water_in_cavity
import sapphire.test


tempdir = sapphire.test.datadir

def test__validate__melt_octadecane__regression(tempdir):
    
    sim = sapphire.benchmarks.melt_octadecane_in_cavity.Simulation(
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
    

def test__validate__freeze_water__regression(tempdir):
    
    endtime = 1.44
    
    sim = sapphire.benchmarks.freeze_water_in_cavity.Simulation(
        element_degree = (1, 2, 2),
        meshsize = 24,
        quadrature_degree = 4,
        time_stencil_size = 3,
        timestep_size = endtime/4.,
        solid_velocity_relaxation_factor = 1.e-12,
        liquidus_smoothing_factor = 1./200.,
        output_directory_path = tempdir)
    
    sim.run(endtime = endtime)
    
    print("Liquid area = {}".format(sim.liquid_area))
    
    assert(abs(sim.liquid_area - 0.69) < 0.01)
    