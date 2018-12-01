import firedrake as fe 
import fempy.models.navier_stokes


def test__verify_convergence_order_via_MMS(
        grid_sizes = (16, 32), tolerance = 0.1):

    class Model(fempy.models.navier_stokes.Model):
    
        def __init__(self, meshsize):
            
            self.meshsize = meshsize
            
            super().__init__()
            
        def init_mesh(self):
            
            self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
        def init_integration_measure(self):

            self.integration_measure = fe.dx(degree = 4)
            
        def init_manufactured_solution(self):
            
            sin, pi = fe.sin, fe.pi
            
            x = fe.SpatialCoordinate(self.mesh)
            
            ihat, jhat = self.unit_vectors()
            
            u = sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
                sin(pi*x[0])*sin(2.*pi*x[1])*jhat
            
            p = -0.5*(u[0]**2 + u[1]**2)
            
            self.manufactured_solution = u, p
        
        def strong_form_residual(self, solution):
        
            grad, dot, div, sym = fe.grad, fe.dot, fe.div, fe.sym
            
            u, p = solution
            
            r_u = grad(u)*u + grad(p) - 2.*div(sym(grad(u)))
            
            r_p = div(u)
            
            return r_u, r_p
        
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = Model,
        expected_order = 2,
        grid_sizes = grid_sizes,
        tolerance = tolerance)
    