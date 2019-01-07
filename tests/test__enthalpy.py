import firedrake as fe 
import fempy.mms
import fempy.models.enthalpy


class VerifiableModel(fempy.models.enthalpy.Model):
    
    def __init__(self, meshsize):
    
        self.meshsize = meshsize
        
        super().__init__()
        
        self.update_initial_values()
        
    def init_mesh(self):
    
        self.mesh = fe.UnitIntervalMesh(self.meshsize)
        
    def init_integration_measure(self):

        self.integration_measure = fe.dx(degree = 4)
        
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

    def update_initial_values(self):
        
        initial_values = fe.interpolate(
            self.manufactured_solution, self.function_space)
        
        self.initial_values.assign(initial_values)
        
def test__verify_spatial_convergence_order_via_mms(
        mesh_sizes = (4, 8, 16, 32),
        timestep_size = 1./256.,
        tolerance = 0.1,
        plot_errors = False,
        plot_solution = False):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableModel,
        parameters = {
            "stefan_number": 0.1,
            "smoothing": 1./32.},
        expected_order = 2,
        mesh_sizes = mesh_sizes,
        timestep_size = timestep_size,
        endtime = 1.,
        tolerance = tolerance,
        plot_errors = plot_errors,
        plot_solution = plot_solution)
        
        
def test__verify_temporal_convergence_order_via_mms(
        meshsize = 256,
        timestep_sizes = (1./16., 1./32., 1./64., 1./128.),
        tolerance = 0.1,
        plot_errors = False,
        plot_solution = False):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = VerifiableModel,
        expected_order = 1,
        meshsize = meshsize,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance,
        plot_errors = plot_errors,
        plot_solution = plot_solution)

        
class SecondOrderVerifiableModel(VerifiableModel):

    def init_initial_values(self):
        
        self.initial_values = [fe.Function(self.function_space) 
            for i in range(2)]
        
    def update_initial_values(self):
        
        initial_values = fe.interpolate(
            self.manufactured_solution, self.function_space)
        
        for iv in self.initial_values:
        
            iv.assign(initial_values)
        
    def init_time_discrete_terms(self):
        """ Gear/BDF2 finite difference scheme 
        with constant time step size. """
        T_np1 = self.solution
        
        T_n = self.initial_values[0]
        
        T_nm1 = self.initial_values[1]
        
        Delta_t = self.timestep_size
        
        T_t = 1./Delta_t*(3./2.*T_np1 - 2.*T_n + 0.5*T_nm1)
        
        phil = self.porosity
        
        phil_t = 1./Delta_t*\
            (3./2.*phil(T_np1) - 2.*phil(T_n) + 0.5*phil(T_nm1))
        
        self.time_discrete_terms = T_t, phil_t
    
    
def test__verify_temporal_convergence_order_via_mms__bdf2(
        meshsize = pow(2, 13),
        timestep_sizes = (1./16., 1./32., 1./64., 1./128.),
        tolerance = 0.1,
        plot_errors = False,
        plot_solution = False):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = SecondOrderVerifiableModel,
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
    