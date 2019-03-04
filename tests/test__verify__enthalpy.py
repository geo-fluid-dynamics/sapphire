import firedrake as fe 
import fempy.mms
import fempy.models.enthalpy


def manufactured_solution(model):
    
    x = fe.SpatialCoordinate(model.mesh)[0]
    
    t = model.time
    
    sin, pi, exp = fe.sin, fe.pi, fe.exp
    
    return 0.5*sin(2.*pi*x)*(1. - 2*exp(-3.*t**2))

    
def test__verify_spatial_convergence__second_order__via_mms(
        mesh_sizes = (4, 8, 16, 32),
        timestep_size = 1./256.,
        tolerance = 0.1):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = fempy.models.enthalpy.Model,
        meshes = [fe.UnitIntervalMesh(size) for size in mesh_sizes],
        model_constructor_kwargs = {
            "quadrature_degree": 4,
            "element_degree": 1,
            "time_stencil_size": 2},
        parameters = {
            "stefan_number": 0.1,
            "smoothing": 1./32.},
        manufactured_solution = manufactured_solution,
        expected_order = 2,
        timestep_size = timestep_size,
        endtime = 1.,
        tolerance = tolerance,
        plot_errors = False,
        plot_solution = False,
        report = False)
        
        
def test__verify_temporal_convergence__first_order__via_mms(
        meshsize = 256,
        timestep_sizes = (1./16., 1./32., 1./64., 1./128.),
        tolerance = 0.1):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = fempy.models.enthalpy.Model,
        mesh = fe.UnitIntervalMesh(meshsize),
        model_constructor_kwargs = {
            "quadrature_degree": 4,
            "element_degree": 1,
            "time_stencil_size": 2},
        parameters = {
            "stefan_number": 0.1,
            "smoothing": 1./32.},
        manufactured_solution = manufactured_solution,
        expected_order = 1,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance,
        plot_errors = False,
        plot_solution = False,
        report = False)

    
def test__verify_temporal_convergence__second_order__via_mms(
        meshsize = 128,
        timestep_sizes = (1./64., 1./128., 1./256.),
        tolerance = 0.3,
        plot_errors = False,
        plot_solution = False):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = fempy.models.enthalpy.Model,
        mesh = fe.UnitIntervalMesh(meshsize),
        model_constructor_kwargs = {
            "quadrature_degree": None,
            "element_degree": 2,
            "time_stencil_size": 3},
        parameters = {
            "stefan_number": 0.1,
            "smoothing": 1./32.},
        manufactured_solution = manufactured_solution,
        expected_order = 2,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance,
        plot_errors = plot_errors,
        plot_solution = plot_solution)
    