import firedrake as fe 
import sapphire.benchmarks.heat_driven_cavity

    
def check_scalar_solution_component(
            solution, 
            component, 
            coordinates, 
            truth_values, 
            relative_tolerance, 
            absolute_tolerance,
            subcomponent = None):
        """Verify the scalar values of a specified solution component.
        
        Args:
            solution (fe.Function): The solution to be verified.
            component (int): Index to a scalar solution component 
                to be verified. The solution is often vector-valued and
                based on a mixed formulation.
            coordinates (List[Tuple[float]]): Spatial coordinates
                to be verified. Each tuple contains a float for each 
                spatial dimension and will be converted to a `fe.Point`.
            truth_values (Tuple[float]): Truth values 
                at each coordinate.
            relative_tolerance (float): Used to assert relative error 
                is not too large.
            absolute_tolerance (float): Used to assert absolute error
                is not too large. This will be used instead of 
                relative error for small values.
        """
        assert(len(truth_values) == len(coordinates))
        
        for i, verified_value in enumerate(truth_values):
            
            values = solution.at(coordinates[i])
            
            value = values[component]
            
            if not(subcomponent == None):
            
                value = value[subcomponent]
            
            absolute_error = abs(value - verified_value)
            
            if abs(verified_value) > absolute_tolerance:
            
                relative_error = absolute_error/verified_value
           
                assert(relative_error < relative_tolerance)
                
            else:
            
                assert(absolute_error < absolute_tolerance)
                
                
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
    check_scalar_solution_component(
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
    