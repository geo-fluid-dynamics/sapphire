import firedrake as fe 
import fempy.models.heat


class Model(fempy.models.heat.Model):
    
    def __init__(self, meshsize):
    
        self.meshsize = meshsize
        
        super().__init__()
        
    def init_mesh(self):
    
        self.mesh = fe.UnitIntervalMesh(self.meshsize)
        
    def init_integration_measure(self):

        self.integration_measure = fe.dx
        
    def strong_form_residual(self, solution):
        
        u = solution
        
        t = self.time
        
        diff, div, grad = fe.diff, fe.div, fe.grad
        
        return diff(u, t) - div(grad(u))
    
    def init_manufactured_solution(self):
        
        x = fe.SpatialCoordinate(self.mesh)[0]
        
        t = self.time
        
        sin, pi, exp = fe.sin, fe.pi, fe.exp
        
        self.manufactured_solution = sin(2.*pi*x)*exp(-pow(t, 2))
    
    def init_initial_values(self):
        
        self.initial_values = fe.interpolate(
            self.manufactured_solution, self.function_space)
    
    def init_solver(self, solver_parameters = {"ksp_type": "cg"}):
        
        self.solver = fe.NonlinearVariationalSolver(
            self.problem, solver_parameters = solver_parameters)

        
def test__verify_spatial_convergence_order_via_mms(
        grid_sizes = (4, 8, 16, 32),
        timestep_size = 1./64.,
        tolerance = 0.1):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = Model,
        expected_order = 2,
        grid_sizes = grid_sizes,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 1.)
        
        
def test__verify_temporal_convergence_order_via_mms(
        meshsize = 256,
        timestep_sizes = (1./4., 1./8., 1./16., 1./32.),
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
        unp1 = self.solution
        
        un = self.initial_values[0]
        
        unm1 = self.initial_values[1]
        
        Delta_t = self.timestep_size
        
        self.time_discrete_terms = 1./Delta_t*(3./2.*unp1 - 2.*un + 0.5*unm1)
        
    def init_manufactured_solution(self):
        
        x = fe.SpatialCoordinate(self.mesh)[0]
        
        t = self.time
        
        sin, pi, exp = fe.sin, fe.pi, fe.exp
        
        self.manufactured_solution = sin(2.*pi*x)*exp(-pow(t, 3))
    
    
def test__verify_temporal_convergence_order_via_mms__bdf2(
        meshsize = 256,
        timestep_sizes = (1./2., 1./4., 1./8., 1./16.),
        tolerance = 0.1):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = SecondOrderModel,
        expected_order = 2,
        meshsize = meshsize,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance)
        
        
class ModelWithWave(Model):
    
    def init_manufactured_solution(self):
        
        x = fe.SpatialCoordinate(self.mesh)[0]
        
        t = self.time
        
        sin, pi, exp = fe.sin, fe.pi, fe.exp
        
        self.manufactured_solution = 0.5*sin(2.*pi*x - pi/4.*(2.*t + 1.))
        
        
def __fails__test_verify_spatial_convergence_order_via_mms_with_wave_solution(
        grid_sizes = (4, 8, 16, 32),
        timestep_size = 1./64.,
        tolerance = 0.1):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = ModelWithWave,
        expected_order = 2,
        grid_sizes = grid_sizes,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 1.)
        