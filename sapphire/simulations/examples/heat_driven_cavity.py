import firedrake as fe
import sapphire.simulations.navier_stokes_boussinesq


def initial_values(sim):

    return sim.solution

    
def dirichlet_boundary_conditions(sim):
    
    W = sim.function_space
    
    return [
        fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
        fe.DirichletBC(W.sub(2), sim.hot_wall_temperature, 1),
        fe.DirichletBC(W.sub(2), sim.cold_wall_temperature, 2)]
        

default_Ra = 1.e6

default_Pr = 0.71

default_Gr = default_Ra/default_Pr

class Simulation(sapphire.simulations.navier_stokes_boussinesq.Simulation):
    
    def __init__(self, *args, 
            meshsize, 
            hot_wall_temperature = 0.5,
            cold_wall_temperature = -0.5,
            grashof_number = default_Gr,
            prandtl_number = default_Pr,
            **kwargs):
        
        self.hot_wall_temperature = fe.Constant(hot_wall_temperature)
    
        self.cold_wall_temperature = fe.Constant(cold_wall_temperature)
        
        super().__init__(
            *args,
            mesh = fe.UnitSquareMesh(meshsize, meshsize),
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            grashof_number = grashof_number,
            prandtl_number = prandtl_number,
            **kwargs)
        