import firedrake as fe 
import fempy.mms
import fempy.models.binary_alloy_enthalpy


class VerifiableModel(fempy.models.binary_alloy_enthalpy.Model):
    
    def __init__(self, meshsize):
    
        self.meshsize = meshsize
        
        super().__init__()
        
    def init_mesh(self):
    
        self.mesh = fe.UnitIntervalMesh(self.meshsize)
        
    def init_integration_measure(self):

        self.integration_measure = fe.dx(degree = 4)
        
    def strong_form_residual(self, solution):
        
        T, Cl = solution
        
        t = self.time
        
        Ste = self.stefan_number
        
        Le = self.lewis_number
        
        Cs = self.solid_concentration
        
        phil = self.porosity(T, Cl)
        
        diff, div, grad = fe.diff, fe.div, fe.grad
        
        r_T = diff(T, t) - div(grad(T)) + 1./Ste*diff(phil, t)
        
        r_Cl = phil*diff(Cl, t) - 1./Le*div(phil*grad(Cl)) + \
            (Cl - Cs)*diff(phil, t)
        
        return r_T, r_Cl
    
    def init_manufactured_solution(self):
        
        x = fe.SpatialCoordinate(self.mesh)[0]
        
        t = self.time
        
        sin, pi, exp = fe.sin, fe.pi, fe.exp
        
        T = 0.5*sin(2.*pi*x)*(1. - 2*exp(-3.*t**2))
        
        Cl = 1. + sin(3.*pi*x)*(2. - exp(-3.*t**2))
        
        self.manufactured_solution = T, Cl

    def init_initial_values(self):
        
        self.initial_values = fe.Function(self.function_space)
        
        for u_m, V in zip(
                self.manufactured_solution, self.function_space):
        
            self.initial_values.assign(fe.interpolate(u_m, V))
            

def test__verify_spatial_convergence_order_via_mms(
        mesh_sizes = (4, 8, 16, 32, 64),
        timestep_size = 1./256.,
        tolerance = 0.1,
        plot_errors = False,
        plot_solution = False):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableModel,
        parameters = {
            "stefan_number": 0.1,
            "lewis_number": 8.,
            "solid_concentration": 0.02,
            "latent_heat_smoothing": 1./32.},
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
        parameters = {
            "stefan_number": 0.1,
            "lewis_number": 80.,
            "solid_concentration": 0.02,
            "latent_heat_smoothing": 1./32.},
        expected_order = 1,
        meshsize = meshsize,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance,
        plot_errors = plot_errors,
        plot_solution = plot_solution)

        
class SecondOrderVerifiableModel(VerifiableModel):

    def init_initial_values(self):
        
        initial_values = fe.interpolate(
            self.manufactured_solution, self.function_space)
        
        self.initial_values = [fe.Function(self.function_space) 
            for i in range(2)]
        
        for iv in self.initial_values:
        
            iv.assign(initial_values)
        
    def init_time_discrete_terms(self):
        """ Gear/BDF2 finite difference scheme 
        with constant time step size. """
        T_np1, Cl_np1 = self.solution
        
        T_n, Cl_n = self.initial_values[0]
        
        T_nm1, Cl_nm1 = self.initial_values[1]
        
        Delta_t = self.timestep_size
        
        T_t = 1./Delta_t*(3./2.*T_np1 - 2.*T_n + 0.5*T_nm1)
        
        Cl_t = 1./Delta_t*(3./2.*Cl_np1 - 2.*Cl_n + 0.5*Cl_nm1)
        
        phil = self.porosity
        
        phil_t = 1./Delta_t*(3./2.*phil(T_np1, Cl_np1) - 
            2.*phil(T_n, Cl_n) + 0.5*phil(T_nm1, Cl_nm1))
        
        self.time_discrete_terms = T_t, Cl_t, phil_t
    
    
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
            "schmidt_number": 8.,
            "solid_concentration": 0.02,
            "latent_heat_smoothing": 1./32},
        expected_order = 2,
        meshsize = meshsize,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance,
        plot_errors = plot_errors,
        plot_solution = plot_solution)
    