import firedrake as fe 
import sapphire.benchmarks.heat_driven_cavity
import sapphire.test


def test__validate_heat_driven_cavity_benchmark():

    sim = sapphire.benchmarks.heat_driven_cavity.Simulation(
        element_degree = (1, 2, 2), meshsize = 40)
    
    sim.solution = sim.solve()
    
    # Verify against the result published in @cite{wang2010comprehensive}.
    Gr = sim.grashof_number.__float__()
    
    Pr = sim.prandtl_number.__float__()
    
    Ra = Gr*Pr
    
    # Check coordinates (0.3499, 0.8499, 0.9999) instead of (0.35, 0.85, 1)
    # because the Function evaluation fails at the exact coordinates.
    # See https://github.com/firedrakeproject/firedrake/issues/1340 
    sapphire.test.check_scalar_solution_component(
        solution = sim.solution,
        component = 1,
        subcomponent = 0,
        coordinates = [(0.5, y) 
            for y in (0., 0.15, 0.34999, 0.5, 0.65, 0.84999, 0.99999)],
        truth_values = [val*Ra**0.5/Pr
            for val in (0.0000, -0.0649, -0.0194, 0.0000, 
                        0.0194, 0.0649, 0.0000)],
        relative_tolerance = 1.e-2,
        absolute_tolerance = 1.e-2*0.0649*Ra**0.5/Pr)
    