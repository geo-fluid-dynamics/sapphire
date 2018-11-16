import firedrake as fe 
import fem


def test__verify_convergence_order_via_MMS():

    class Model(fem.models.navier_stokes.Model):
    
        def __init__(self, gridsize):
            
            self.gridsize = gridsize
            
            super().__init__()
            
            self.integration_measure = fe.dx(degree = 4)
            
        def init_mesh(self):
            
            self.mesh = fe.UnitSquareMesh(self.gridsize, self.gridsize)
            
        def strong_form_residual(self):
        
            grad, dot, div, sym = fe.grad, fe.dot, fe.div, fe.sym
            
            u, p = self.manufactured_solution()
            
            r_u = grad(u)*u + grad(p) - 2.*div(sym(grad(u)))
            
            r_p = div(u)
            
            return r_u, r_p
        
        def manufactured_solution(self):
            
            sin, pi = fe.sin, fe.pi
            
            x = fe.SpatialCoordinate(self.mesh)
            
            ihat, jhat = self.unit_vectors()
            
            u = sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
                sin(pi*x[0])*sin(2.*pi*x[1])*jhat
            
            p = -0.5*(u[0]**2 + u[1]**2)
            
            return u, p
    
    fem.mms.verify_order_of_accuracy(
        Model = Model,
        expected_spatial_order = 2,
        grid_sizes = (8, 16, 32),
        tolerance = 0.1)
    