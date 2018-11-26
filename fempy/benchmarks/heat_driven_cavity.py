import firedrake as fe
import fempy.models.navier_stokes_boussinesq


class Model(fempy.models.navier_stokes_boussinesq.Model):
    
    def __init__(self, meshsize):
        
        self.meshsize = meshsize
        
        self.hot_wall_temperature = fe.Constant(0.5)
    
        self.cold_wall_temperature = fe.Constant(-0.5)
        
        super().__init__()
        
        self.rayleigh_number.assign(1.e6)
        
        self.prandtl_number.assign(0.71)
        
    def init_mesh(self):
    
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def init_dirichlet_boundary_conditions(self):
    
        W = self.function_space
        
        self.dirichlet_boundary_conditions = [
            fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
            fe.DirichletBC(W.sub(2), self.hot_wall_temperature, 1),
            fe.DirichletBC(W.sub(2), self.cold_wall_temperature, 2)]
            