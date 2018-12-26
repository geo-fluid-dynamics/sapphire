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
        
        Cs = self.solid_concentration
        
        Cl = 0.5 + Cs - T
        
        self.manufactured_solution = T, Cl

    def init_initial_values(self):
        
        self.initial_values = fe.Function(self.function_space)
        
        for u_m, V in zip(
                self.manufactured_solution, self.function_space):
        
            self.initial_values.assign(fe.interpolate(u_m, V))
            

def fails__test__verify_spatial_convergence_order_via_mms(
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
        