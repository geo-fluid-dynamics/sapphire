import firedrake as fe 
import fempy.mms
import fempy.models.enthalpy_phasechange


class Model(fempy.models.enthalpy_phasechange.Model):
    
    def __init__(self, meshsize):
    
        self.meshsize = meshsize
        
        super().__init__()
        
        self.stefan_number.assign(0.1)
        
        self.phase_interface_smoothing.assign(1./32.)
        
    def init_mesh(self):
    
        self.mesh = fe.UnitIntervalMesh(self.meshsize)
        
    def init_integration_measure(self):

        self.integration_measure = fe.dx(degree = 4)
        
    def strong_form_residual(self, solution):
        
        theta = solution
        
        t = self.time
        
        Ste = self.stefan_number
        
        phi = self.semi_phasefield
        
        diff, div, grad = fe.diff, fe.div, fe.grad
        
        return diff(theta, t) - div(grad(theta)) - 1./Ste*diff(phi(theta), t)
    
    def init_manufactured_solution(self):
        
        x = fe.SpatialCoordinate(self.mesh)[0]
        
        t = self.time
        
        sin, pi, exp = fe.sin, fe.pi, fe.exp
        
        self.manufactured_solution = 0.5*sin(2.*pi*x)*(1. - 2*exp(-3.*t**2))

    def init_initial_values(self):
        
        self.initial_values = [fe.interpolate(
            self.manufactured_solution, self.function_space),]
        

def test__verify_spatial_convergence_order_via_mms(
        grid_sizes = (4, 8, 16, 32),
        timestep_size = 1./256.,
        tolerance = 0.1):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = Model,
        expected_order = 2,
        grid_sizes = grid_sizes,
        timestep_size = timestep_size,
        endtime = 1.,
        tolerance = tolerance)
        
        
def test__verify_temporal_convergence_order_via_mms(
        meshsize = 256,
        timestep_sizes = (1./16., 1./32., 1./64., 1./128.),
        tolerance = 0.1):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = Model,
        expected_order = 1,
        meshsize = meshsize,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance)

        
class SecondOrderModel(Model):

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
        thetanp1 = self.solution
        
        thetan = self.initial_values[0]
        
        thetanm1 = self.initial_values[1]
        
        Delta_t = self.timestep_size
        
        theta_t = 1./Delta_t*(3./2.*thetanp1 - 2.*thetan + 0.5*thetanm1)
        
        phi = self.semi_phasefield
        
        phi_t = 1./Delta_t*\
            (3./2.*phi(thetanp1) - 2.*phi(thetan) + 0.5*phi(thetanm1))
        
        self.time_discrete_terms = theta_t, phi_t
    
    
def test__verify_temporal_convergence_order_via_mms__bdf2(
        meshsize = pow(2, 13),
        timestep_sizes = (1./16., 1./32., 1./64., 1./128.),
        tolerance = 0.1):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = SecondOrderModel,
        expected_order = 2,
        meshsize = meshsize,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance)
    