import firedrake as fe 
import fempy.models.laplace


def test__verify_convergence_order_via_mms(
        grid_sizes = (16, 32), tolerance = 0.1):

    class Model(fempy.models.laplace.Model):
    
        def __init__(self, meshsize):
            
            self.meshsize = meshsize
            
            super().__init__()
            
            self.integration_measure = fe.dx(degree = 2)
            
        def init_mesh(self):
            
            self.mesh = fe.UnitIntervalMesh(self.meshsize)
            
        def init_manufactured_solution(self):
            
            sin, pi = fe.sin, fe.pi
            
            x = fe.SpatialCoordinate(self.mesh)[0]
            
            self.manufactured_solution = sin(2.*pi*x)
            
        def strong_form_residual(self, solution):
        
            div, grad, = fe.div, fe.grad
            
            u = solution
            
            return div(grad(u))
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = Model,
        expected_order = 2,
        grid_sizes = grid_sizes,
        tolerance = tolerance)
    