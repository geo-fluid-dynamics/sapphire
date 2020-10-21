import sapphire.simulations.examples.melt_gallium


def test__melt_gallium__regression(tmpdir):
    
    sim = sapphire.simulations.examples.melt_gallium.Simulation(
        taylor_hood_pressure_degree = 1,
        temperature_degree = 2,
        mesh_dimensions = (20, 40),
        quadrature_degree = 4,
        time_stencil_size = 3,
        timestep_size = 0.0125,
        liquidus_smoothing_factor = 0.05,
        solid_velocity_relaxation_factor = 1.e-10,
        output_directory_path = tmpdir)
    
    sim.run(endtime = 0.3)
    
    print("Liquid area = {}".format(sim.liquid_area))
    
    assert(round(sim.liquid_area, 2) == 0.17)
    