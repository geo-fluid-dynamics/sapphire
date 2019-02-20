import firedrake as fe 
import fempy.mms
import fempy.models.enthalpy


class VerifiableModel(fempy.models.enthalpy.Model):
    
    def __init__(self, 
            quadrature_degree,
            spatial_order,
            temporal_order,
            meshsize):
    
        self.meshsize = meshsize
        
        super().__init__(
            quadrature_degree = quadrature_degree,
            spatial_order = spatial_order,
            temporal_order = temporal_order)
        
    def init_mesh(self):
    
        self.mesh = fe.UnitIntervalMesh(self.meshsize)
        
    def strong_form_residual(self, solution):
        
        T = solution
        
        t = self.time
        
        Ste = self.stefan_number
        
        phil = self.porosity
        
        diff, div, grad = fe.diff, fe.div, fe.grad
        
        return diff(T, t) - div(grad(T)) + 1./Ste*diff(phil(T), t)
    
    def init_manufactured_solution(self):
        
        x = fe.SpatialCoordinate(self.mesh)[0]
        
        t = self.time
        
        sin, pi, exp = fe.sin, fe.pi, fe.exp
        
        self.manufactured_solution = 0.5*sin(2.*pi*x)*(1. - 2*exp(-3.*t**2))
    
    
def test__verify_spatial_convergence_order_via_mms(
        mesh_sizes = (4, 8, 16, 32),
        timestep_size = 1./256.,
        tolerance = 0.1):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableModel,
        constructor_kwargs = {
            "quadrature_degree": 4,
            "spatial_order": 2,
            "temporal_order": 1},
        parameters = {
            "stefan_number": 0.1,
            "smoothing": 1./32.},
        expected_order = 2,
        mesh_sizes = mesh_sizes,
        timestep_size = timestep_size,
        endtime = 1.,
        tolerance = tolerance,
        plot_errors = False,
        plot_solution = False,
        report = False)
        
        
def test__verify_temporal_convergence_order_via_mms(
        meshsize = 256,
        timestep_sizes = (1./16., 1./32., 1./64., 1./128.),
        tolerance = 0.1):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = VerifiableModel,
        constructor_kwargs = {
            "quadrature_degree": 4,
            "spatial_order": 2,
            "temporal_order": 1},
        parameters = {
            "stefan_number": 0.1,
            "smoothing": 1./32.},
        expected_order = 1,
        meshsize = meshsize,
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
        Model = VerifiableModel,
        constructor_kwargs = {
            "quadrature_degree": None,
            "spatial_order": 3,
            "temporal_order": 2},
        parameters = {
            "stefan_number": 0.1,
            "smoothing": 1./32.},
        expected_order = 2,
        meshsize = meshsize,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance,
        plot_errors = plot_errors,
        plot_solution = plot_solution)
    