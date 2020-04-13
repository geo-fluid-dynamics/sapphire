import firedrake as fe
import sapphire.simulations.navier_stokes
import typing


def initial_values(sim):

    return sim.solution

    
def dirichlet_boundary_conditions(sim):
    
    W = sim.function_space
    
    return [
        fe.DirichletBC(W.sub(0), fe.Constant((0., 0.)), (1, 2, 3)),
        fe.DirichletBC(W.sub(0), fe.Constant((1., 0.)), 4)]
        
        
class Simulation(sapphire.simulations.navier_stokes.Simulation):
    
    def __init__(self, *args, 
            mesh: fe.UnitSquareMesh = None,
            reynolds_number = 100.,
            **kwargs):
        
        if mesh is None:
        
            mesh = fe.UnitSquareMesh(50, 50)
            
        super().__init__(
            *args,
            mesh = mesh,
            reynolds_number = reynolds_number,
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        