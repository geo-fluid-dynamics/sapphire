import firedrake as fe 
import fem


def test__verify_convergence_order_via_mms(
        grid_sizes = (16, 32), timestep_sizes = (1., 1./2., 1./4.), tolerance = 0.1):
    
    class Model(fem.models.heat.Model):
    
        def __init__(self, gridsize = 4):
        
            self.gridsize = gridsize
            
            super().__init__()
            
            self.integration_measure = fe.dx(degree = 2)
            
            self.thermal_diffusivity.assign(3.)
            
            self.timestep_size.assign(max(timestep_sizes))
            
            self.time = 1.
            
        def init_mesh(self):
        
            self.mesh = fe.UnitSquareMesh(self.gridsize, self.gridsize)
            
        def strong_form_residual(self, solution):
            
            alpha = self.thermal_diffusivity
            
            u = solution
            
            t = self.ufl_time
            
            diff, div, grad = fe.diff, fe.div, fe.grad
            
            return diff(u, t) - alpha*div(grad(u))
        
        def manufactured_solution(self):
            
            x = fe.SpatialCoordinate(self.mesh)
            
            t = self.ufl_time
            
            sin, pi, exp = fe.sin, fe.pi, fe.exp
            
            return sin(2.*pi*x[0])*sin(pi*x[1])*exp(-t)
        
        def init_solution(self):
        
            super().init_solution()        
        
            u = fe.Expression(self.manufactured_solution())
            
            u.t = self.time
            
            self.set_initial_values(u)
        
    fem.mms.verify_spatial_order_of_accuracy(
        Model = Model,
        expected_order = 2,
        grid_sizes = (8, 16, 32),
        tolerance = tolerance)
    
    fem.mms.verify_temporal_order_of_accuracy(
        Model = Model,
        expected_order = 1,
        gridsize = max(grid_sizes),
        endtime = 1.,
        timestep_sizes = (1., 1./2., 1./4.),
        tolerance = tolerance)
