"""Helper functions for test code"""

    
def check_scalar_solution_component(
        solution, 
        component, 
        coordinates, 
        expected_values, 
        absolute_tolerances,
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
        expected_values (Tuple[float]): Truth values 
            at each coordinate.
        absolute_tolerances (Tuple[float]): Used to assert absolute error 
            is not too large. Specify a tolerance for each value.
    """
    assert(len(expected_values) == len(coordinates))
    
    indices = range(len(expected_values))
    
    for i, expected_value, tolerance in zip(
            indices, expected_values, absolute_tolerances):
        
        values = solution.at(coordinates[i])
        
        value = values[component]
        
        if not(subcomponent == None):
        
            value = value[subcomponent]
        
        print("Expected {} and found {}.".format(expected_value, value))
        
        absolute_error = abs(value - expected_value)
        
        assert absolute_error <= tolerance
        