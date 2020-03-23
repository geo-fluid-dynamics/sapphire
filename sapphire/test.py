from pytest import fixture
from distutils import dir_util
import os


@fixture
def datadir(tmpdir, request):
    """    
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    
    Copied from https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    
    Dear future developers:
    - Should this be done with pathlib now?
    """
    filename = request.module.__file__
    
    testdir, _ = os.path.splitext(filename)

    if os.path.isdir(testdir):
    
        dir_util.copy_tree(testdir, str(tmpdir))

    return tmpdir
    
    
def check_scalar_solution_component(
        solution, 
        component, 
        coordinates, 
        expected_values, 
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
        expected_values (Tuple[float]): Truth values 
            at each coordinate.
        relative_tolerance (float): Used to assert relative error 
            is not too large.
        absolute_tolerance (float): Used to assert absolute error
            is not too large. This will be used instead of 
            relative error for small values.
    """
    assert(len(expected_values) == len(coordinates))
    
    for i, verified_value in enumerate(expected_values):
        
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
            