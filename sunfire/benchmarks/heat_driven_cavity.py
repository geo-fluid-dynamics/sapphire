import firedrake as fe
import sunfire.models.navier_stokes_boussinesq


def initial_values(model):

    return model.solution

def dirichlet_boundary_conditions(model):
    
    W = model.function_space
    
    return [
        fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
        fe.DirichletBC(W.sub(2), model.hot_wall_temperature, 1),
        fe.DirichletBC(W.sub(2), model.cold_wall_temperature, 2)]

        
class Model(sunfire.models.navier_stokes_boussinesq.Model):
    
    def __init__(self, *args, meshsize, **kwargs):
        
        self.hot_wall_temperature = fe.Constant(0.5)
    
        self.cold_wall_temperature = fe.Constant(-0.5)
        
        super().__init__(
            *args,
            mesh = fe.UnitSquareMesh(meshsize, meshsize),
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        
        Ra = 1.e6
        
        Pr = 0.71
        
        self.grashof_number = self.grashof_number.assign(Ra/Pr)
        
        self.prandtl_number = self.prandtl_number.assign(Pr)
        