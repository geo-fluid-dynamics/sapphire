import firedrake as fe
import sapphire.simulations.navier_stokes
import typing


class Simulation(sapphire.simulations.navier_stokes.Simulation):
    
    def __init__(self, *args, 
            mesh_dimensions = (50, 50),
            reynolds_number = 100,
            **kwargs):
            
        if "solution" not in kwargs:
            
            kwargs["mesh"] = fe.UnitSquareMesh(*mesh_dimensions)
            
        super().__init__(
            *args,
            reynolds_number = reynolds_number,
            **kwargs)
    
    def dirichlet_boundary_conditions(sim):
    
        W = sim.solution_space
        
        return [
            fe.DirichletBC(
                sim.solution_subspaces["u"],
                fe.Constant((0, 0)),
                (1, 2, 3)),
            fe.DirichletBC(
                sim.solution_subspaces["u"],
                fe.Constant((1, 0)),
                4)]
            