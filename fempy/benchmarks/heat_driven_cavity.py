import firedrake as fe
import fempy.models.navier_stokes_boussinesq


class Model(fempy.models.navier_stokes_boussinesq.Model):
    
    def __init__(self, quadrature_degree, spatial_order, meshsize):
        
        self.meshsize = meshsize
        
        self.hot_wall_temperature = fe.Constant(0.5)
    
        self.cold_wall_temperature = fe.Constant(-0.5)
        
        super().__init__(
            quadrature_degree = quadrature_degree,
            spatial_order = spatial_order)
        
        Ra = 1.e6
        
        Pr = 0.71
        
        self.grashof_number.assign(Ra/Pr)
        
        self.prandtl_number.assign(Pr)
        
    def init_mesh(self):
    
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def init_dirichlet_boundary_conditions(self):
    
        W = self.function_space
        
        self.dirichlet_boundary_conditions = [
            fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
            fe.DirichletBC(W.sub(2), self.hot_wall_temperature, 1),
            fe.DirichletBC(W.sub(2), self.cold_wall_temperature, 2)]
            