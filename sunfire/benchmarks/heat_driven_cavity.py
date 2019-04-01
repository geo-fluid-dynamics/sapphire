import firedrake as fe
import sunfire.simulations.navier_stokes_boussinesq


def initial_values(sim):

    return sim.solution

def dirichlet_boundary_conditions(sim):
    
    W = sim.function_space
    
    return [
        fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
        fe.DirichletBC(W.sub(2), sim.hot_wall_temperature, 1),
        fe.DirichletBC(W.sub(2), sim.cold_wall_temperature, 2)]

        
class Simulation(sunfire.simulations.navier_stokes_boussinesq.Simulation):
    
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
        