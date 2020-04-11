import firedrake as fe
import sapphire.simulations.navier_stokes


def initial_values(sim):

    return sim.solution

    
def dirichlet_boundary_conditions(sim):
        
    W = sim.function_space
    
    return [
        fe.DirichletBC(W.sub(0), fe.Constant((0., 0.)), (1, 2, 3)),
        fe.DirichletBC(W.sub(0), fe.Constant((1., 0.)), 4)]
        
        
class Simulation(sapphire.simulations.navier_stokes.Simulation):
    
    def __init__(self, *args, 
            horizontal_cellcount, 
            vertical_cellcount,
            **kwargs):
        
        super().__init__(
            *args,
            mesh = fe.UnitSquareMesh(
                horizontal_cellcount, vertical_cellcount),
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        