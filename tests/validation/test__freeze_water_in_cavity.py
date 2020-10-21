import firedrake as fe 
import sapphire.simulations.examples.freeze_water_in_cavity


def test__freeze_water(tmpdir):
    
    endtime = 1.44
    
    sim = sapphire.simulations.examples.freeze_water_in_cavity.Simulation(
        taylor_hood_pressure_degree = 1,
        temperature_degree = 2,
        mesh_dimensions = (24, 24),
        quadrature_degree = 4,
        time_stencil_size = 3,
        timestep_size = endtime/4.,
        solid_velocity_relaxation_factor = 1.e-12,
        liquidus_smoothing_factor = 0.005,
        output_directory_path = tmpdir)
    
    sim.run(endtime = endtime, write_plots = True)
    
    print("Liquid area = {}".format(sim.liquid_area))
    
    assert(round(sim.liquid_area, 2) == 0.70)
    