import firedrake as fe 
import fem


def test__verify_convergence_order_via_mms(
        grid_sizes = (16, 32), tolerance = 0.1):

    class Model(fem.models.laplace.Model):
    
        def __init__(self, gridsize):
            
            self.gridsize = gridsize
            
            super().__init__()
            
            self.integration_measure = fe.dx(degree = 2)
            
        def init_mesh(self):
            
            self.mesh = fe.UnitSquareMesh(self.gridsize, self.gridsize)
            
        def strong_form_residual(self, solution):
        
            div, grad, = fe.div, fe.grad
            
            u = solution
            
            return div(grad(u))
        
        def manufactured_solution(self):
            
            sin, pi = fe.sin, fe.pi
            
            x = fe.SpatialCoordinate(self.mesh)
            
            return sin(2.*pi*x[0])*sin(pi*x[1])
    
    fem.mms.verify_spatial_order_of_accuracy(
        Model = Model,
        expected_order = 2,
        grid_sizes = grid_sizes,
        tolerance = tolerance)
    