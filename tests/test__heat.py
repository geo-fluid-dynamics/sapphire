import firedrake as fe 
import fem


class Model(fem.models.heat.Model):
    
    def __init__(self, gridsize):
    
        self.gridsize = gridsize
        
        super().__init__()
        
        self.thermal_diffusivity.assign(3.)
        
    def init_mesh(self):
    
        self.mesh = fe.UnitSquareMesh(self.gridsize, self.gridsize)
        
    def init_integration_measure(self):

            self.integration_measure = fe.dx(degree = 2)
        
    def strong_form_residual(self, solution):
        
        alpha = self.thermal_diffusivity
        
        u = solution
        
        t = self.time
        
        diff, div, grad = fe.diff, fe.div, fe.grad
        
        return diff(u, t) - alpha*div(grad(u))
    
    def init_manufactured_solution(self):
        
        x = fe.SpatialCoordinate(self.mesh)
        
        t = self.time
        
        sin, pi, exp = fe.sin, fe.pi, fe.exp
        
        self.manufactured_solution = \
            sin(2.*pi*x[0])*sin(pi*x[1])*exp(-pow(t, 2))

            
def test__verify_spatial_convergence_order_via_mms(
        grid_sizes = (16, 32), 
        timestep_size = 1./32.,
        tolerance = 0.1,
        quadrature_degree = 2):
    
    fem.mms.verify_spatial_order_of_accuracy(
        Model = Model,
        expected_order = 2,
        grid_sizes = grid_sizes,
        tolerance = tolerance,
        quadrature_degree = quadrature_degree,
        timestep_size = timestep_size)
        
        
def test__verify_temporal_convergence_order_via_mms(
        gridsize = 32, 
        timestep_sizes = (1./16., 1./32.),
        tolerance = 0.1,
        quadrature_degree = 2):
    
    fem.mms.verify_temporal_order_of_accuracy(
        Model = Model,
        expected_order = 1,
        gridsize = gridsize,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance,
        quadrature_degree = quadrature_degree)
