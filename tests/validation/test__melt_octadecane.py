import sapphire.simulations.examples.melt_octadecane


def test__melt_octadecane(tmpdir):
    
    sim = sapphire.simulations.examples.melt_octadecane.Simulation(
        taylor_hood_pressure_degree = 1,
        temperature_degree = 1,
        mesh_dimensions = (24, 24),
        quadrature_degree = 4,
        time_stencil_size = 3,
        timestep_size = 20.,
        liquidus_smoothing_factor = 0.005,
        solid_velocity_relaxation_factor = 1.e-12,
        output_directory_path = tmpdir)
    
    sim.run(endtime = 80.)
    
    print("Liquid area = {}".format(sim.liquid_area))
    
    assert(round(sim.liquid_area, 2) == 0.47)
    