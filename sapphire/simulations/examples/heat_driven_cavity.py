import firedrake as fe
import sapphire.simulations.navier_stokes_boussinesq
import typing


class Simulation(sapphire.simulations.navier_stokes_boussinesq.Simulation):
    
    def __init__(self, *args,
            mesh_dimensions = (20, 20),
            hotwall_temperature = 0.5,
            coldwall_temperature = -0.5,
            grashof_number = 1.e6/0.71,
            prandtl_number = 0.71,
            **kwargs):
        
        if "solution" not in kwargs:
            
            kwargs["mesh"] = fe.UnitSquareMesh(*mesh_dimensions)
            
        self.hotwall_id = 1
        
        self.coldwall_id = 2
        
        self.hotwall_temperature = fe.Constant(hotwall_temperature)
    
        self.coldwall_temperature = fe.Constant(coldwall_temperature)
        
        super().__init__(
            *args,
            grashof_number = grashof_number,
            prandtl_number = prandtl_number,
            **kwargs)
    
    def dirichlet_boundary_conditions(self):
        
        W = self.solution.function_space()
        
        d = self.solution.function_space().mesh().geometric_dimension()
        
        return [
            fe.DirichletBC(
                W.sub(1), (0.,)*d, "on_boundary"),
            fe.DirichletBC(W.sub(2), self.hotwall_temperature, self.hotwall_id),
            fe.DirichletBC(W.sub(2), self.coldwall_temperature, self.coldwall_id)]
            