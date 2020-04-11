""" Validate against steady state heat-driven cavity benchmark.

Comparing to data published in 

    @article{wang2010comprehensive,
        author = {Ghia, Urmila and Ghia, K.N and Shin, C.T},
        year = {1982},
        month = {12},
        pages = {387-411},
        title = {High-Re solutions for incompressible flow using the 
        Navier-Stokes equations and a multigrid method},
        volume = {48},
        journal = {Journal of Computational Physics},
        doi = {10.1016/0021-9991(82)90058-4}
    }
"""
import firedrake as fe 
import sapphire.simulations.examples.heat_driven_cavity
import sapphire.test


def test__validate_heat_driven_cavity_benchmark():

    sim = sapphire.simulations.examples.heat_driven_cavity.Simulation(
        element_degree = (1, 2, 2), meshsize = 40)
    
    sim.solution = sim.solve()
    
    Gr = sim.grashof_number.__float__()
    
    Pr = sim.prandtl_number.__float__()
    
    Ra = Gr*Pr
    
    # Check coordinates (0.3499, 0.8499) instead of (0.35, 0.85)
    # because the Function evaluation fails at the exact coordinates.
    # See https://github.com/firedrakeproject/firedrake/issues/1340 
    sapphire.test.check_scalar_solution_component(
        solution = sim.solution,
        component = 1,
        subcomponent = 0,
        coordinates = [(0.5, y) 
            for y in (0., 0.15, 0.34999, 0.5, 0.65, 0.84999)],
        expected_values = [val*Ra**0.5/Pr
            for val in (0.0000, -0.0649, -0.0194, 0.0000, 
                        0.0194, 0.0649)],
        absolute_tolerances = [val*Ra**0.5/Pr 
            for val in (1.e-12, 0.001, 0.001, 1.e-12, 0.001, 0.001)])
    