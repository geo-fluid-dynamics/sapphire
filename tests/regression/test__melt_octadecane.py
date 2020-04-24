import firedrake as fe
import sapphire.simulations.examples.melt_octadecane
import sapphire.test


tempdir = sapphire.test.datadir

def test__validate__melt_octadecane__regression(tempdir):
    
    sim = sapphire.simulations.examples.melt_octadecane.Simulation(
        element_degree = (1, 2, 1),
        mesh = fe.UnitSquareMesh(24, 24),
        quadrature_degree = 4,
        time_stencil_size = 3,
        timestep_size = 20.,
        liquidus_smoothing_factor = 0.005,
        viscosity_solid_to_liquid_ratio = 1.e8,
        output_directory_path = tempdir)
    
    sim.run(endtime = 80.)
    
    print("Liquid area = {}".format(sim.liquid_area))
    
    assert(abs(sim.liquid_area - 0.46) < 0.01)
    