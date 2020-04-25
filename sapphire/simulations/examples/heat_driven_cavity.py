import firedrake as fe
import sapphire.simulations.navier_stokes_boussinesq
import typing


class Simulation(sapphire.simulations.navier_stokes_boussinesq.Simulation):
    
    def __init__(self, *args, 
            mesh: typing.Union[fe.UnitSquareMesh, fe.UnitCubeMesh] = None,
            hotwall_temperature = 0.5,
            coldwall_temperature = -0.5,
            grashof_number = 1.e6/0.71,
            prandtl_number = 0.71,
            **kwargs):
        
        if mesh is None:
            
            mesh = fe.UnitSquareMesh(40, 40)
            
        self.hotwall_temperature = fe.Constant(hotwall_temperature)
    
        self.coldwall_temperature = fe.Constant(coldwall_temperature)
        
        super().__init__(
            *args,
            mesh = mesh,
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            grashof_number = grashof_number,
            prandtl_number = prandtl_number,
            **kwargs)


def initial_values(sim):

    return sim.solution


hotwall_id, coldwall_id = 1, 2
    
def dirichlet_boundary_conditions(sim):
    
    W = sim.function_space
    
    return [
        fe.DirichletBC(
            W.sub(1), (0.,)*sim.mesh.geometric_dimension(), "on_boundary"),
        fe.DirichletBC(W.sub(2), sim.hotwall_temperature, hotwall_id),
        fe.DirichletBC(W.sub(2), sim.coldwall_temperature, coldwall_id)]
