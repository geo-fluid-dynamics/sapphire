import firedrake as fe 
import fempy.models.unsteady_navier_stokes


class Model(fempy.models.unsteady_navier_stokes.Model):
    
    def __init__(self,
            quadrature_degree, spatial_order, temporal_order, meshsize):
        
        self.meshsize = meshsize
        
        super().__init__(
            quadrature_degree = quadrature_degree,
            spatial_order = spatial_order,
            temporal_order = temporal_order)
        
    def init_mesh(self):
        
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
    
    def init_manufactured_solution(self):
        
        exp, sin, pi = fe.exp, fe.sin, fe.pi
        
        x = fe.SpatialCoordinate(self.mesh)
        
        t = self.time
        
        ihat, jhat = self.unit_vectors()
        
        u = exp(t)*(sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
            sin(pi*x[0])*sin(2.*pi*x[1])*jhat)
        
        p = -0.5*(u[0]**2 + u[1]**2)
        
        self.manufactured_solution = u, p
    
    def strong_form_residual(self, solution):
    
        diff, grad, div, sym = fe.diff, fe.grad, fe.div, fe.sym
        
        u, p = solution
        
        t = self.time
        
        r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(sym(grad(u)))
        
        r_p = div(u)
        
        return r_u, r_p

        
def test__verify_spatial_convergence_order_via_mms(
        mesh_sizes = (3, 6, 12, 24),
        timestep_size = 1./32.,
        tolerance = 0.3):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = Model,
        constructor_kwargs = {
            "quadrature_degree": 4, "spatial_order": 2, "temporal_order": 1},
        expected_order = 2,
        mesh_sizes = mesh_sizes,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 1.)
    
 
def test__verify_temporal_convergence_order_via_mms(
        meshsize = 32,
        timestep_sizes = (1./2., 1./4., 1./8.),
        tolerance = 0.1):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = Model,
        constructor_kwargs = {
            "quadrature_degree": 4, "spatial_order": 2, "temporal_order": 1},
        expected_order = 1,
        meshsize = meshsize,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance)
    