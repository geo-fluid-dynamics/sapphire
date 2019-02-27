import firedrake as fe 
import fempy.models.navier_stokes


class VerifiableModel(fempy.models.navier_stokes.Model):
    
        def __init__(self, quadrature_degree, spatial_order, meshsize):
            
            self.meshsize = meshsize
            
            super().__init__(
                quadrature_degree = quadrature_degree,
                spatial_order = spatial_order)
            
        def init_mesh(self):
            
            self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
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
            

def test__verify_convergence_order_via_mms(
        mesh_sizes = (16, 32), tolerance = 0.1):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableModel,
        constructor_kwargs = {"quadrature_degree": 4, "spatial_order": 2},
        expected_order = 2,
        mesh_sizes = mesh_sizes,
        tolerance = tolerance)
    