import firedrake as fe 
import fem


class Model(fem.models.enthalpy_phasechange.Model):
    
    def __init__(self, gridsize):
    
        self.gridsize = gridsize
        
        super().__init__()
        
        self.stefan_number.assign(0.1)
        
        self.phase_interface_smoothing.assign(1./32.)
        
    def init_mesh(self):
    
        self.mesh = fe.UnitIntervalMesh(self.gridsize)
        
    def init_integration_measure(self):

        self.integration_measure = fe.dx(degree = 4)
        
    def strong_form_residual(self, solution):
        
        theta = solution
        
        t = self.time
        
        Ste = self.stefan_number
        
        phi = self.semi_phasefield
        
        diff, div, grad = fe.diff, fe.div, fe.grad
        
        return diff(theta, t) - 1./Ste*diff(phi(theta), t) - div(grad(theta))
    
    def init_manufactured_solution(self):
        
        x = fe.SpatialCoordinate(self.mesh)[0]
        
        t = self.time
        
        sin, pi = fe.sin, fe.pi
        
        self.manufactured_solution = 0.5*sin(2.*pi*x - pow(t, 2) - pi/4.)

        
def test__verify_spatial_convergence_order_via_mms(
        grid_sizes = (4, 8, 16, 32),
        timestep_size = 1./64.,
        tolerance = 0.1):
    
    fem.mms.verify_spatial_order_of_accuracy(
        Model = Model,
        expected_order = 2,
        grid_sizes = grid_sizes,
        timestep_size = timestep_size,
        endtime = 1.,
        tolerance = tolerance,
        plot_solutions = True)
        
        
def test__verify_temporal_convergence_order_via_mms(
        gridsize = 128,
        timestep_sizes = (1./4., 1./8., 1./16., 1./32.),
        tolerance = 0.1):
    
    fem.mms.verify_temporal_order_of_accuracy(
        Model = Model,
        expected_order = 1,
        gridsize = gridsize,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance,
        plot_solutions = True)
