import firedrake as fe 
import fem


def test__verify_convergence_order_via_mms():

    class Model(fem.models.laplace.Model):
    
        def __init__(self, gridsize):
            
            self.gridsize = gridsize
            
            super().__init__()
            
            self.integration_measure = fe.dx(degree = 2)
            
        def init_mesh(self):
            
            self.mesh = fe.UnitSquareMesh(self.gridsize, self.gridsize)
            
        def strong_form_residual(self):
        
            div, grad, = fe.div, fe.grad
            
            u = self.manufactured_solution()
            
            return div(grad(u))
        
        def manufactured_solution(self):
            
            sin, pi = fe.sin, fe.pi
            
            x = fe.SpatialCoordinate(self.mesh)
            
            return sin(2.*pi*x[0])*sin(pi*x[1])
    
    fem.mms.verify_order_of_accuracy(
        Model = Model,
        expected_spatial_order = 2,
        grid_sizes = (8, 16, 32),
        tolerance = 0.1)
    
    
if __name__ == "__main__":

    print("Using Python " + sys.version)

    print("Using " + fe.__name__ + "-" + fe.__version__)
    
    test__verify_convergence_order_via_mms()
