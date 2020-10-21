""" Validate against steady state lid-driven cavity benchmark.

Comparing to data published in 

    @article{ghia1982high-re,
        author = {Ghia, Urmila and Ghia, K.N and Shin, C.T},
        year = {1982},
        month = {12},
        pages = {387-411},
        title = {High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method1},
        volume = {48},
        journal = {Journal of Computational Physics},
        doi = {10.1016/0021-9991(82)90058-4}
    }
"""
import firedrake as fe 
import sapphire.simulations.examples.lid_driven_cavity
import tests.validation.helpers


def test__lid_driven_cavity(tmpdir):
    
    sim = sapphire.simulations.examples.lid_driven_cavity.Simulation(
        reynolds_number = 100.,
        mesh_dimensions = (50, 50),
        taylor_hood_pressure_degree = 1,
        output_directory_path = tmpdir)
    
    sim.solution = sim.solve()
    
    sim.write_outputs(headers = True)
    
    tests.validation.helpers.check_scalar_solution_component(
        solution = sim.solution,
        component = sim.fieldnames.index('u'),
        subcomponent = 0,
        coordinates = [(0.5, y) 
            for y in (0.9766, 0.1016, 0.0547, 0.0000)],
        expected_values = (0.8412, -0.0643, -0.0372, 0.0000),
        absolute_tolerances = (0.0025, 0.0015, 0.001, 1.e-12))
        