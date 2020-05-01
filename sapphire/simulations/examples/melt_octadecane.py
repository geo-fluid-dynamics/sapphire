""" Octadecane melting simulation

based on

    @article{danaila2014newton,
        title={A {N}ewton method with adaptive finite elements 
            for solving phase-change problems with natural convection},
        author={Danaila, Ionut and Moglan, Raluca and Hecht, 
            Fr{\'e}d{\'e}ric and Le Masson, St{\'e}phane},
        journal={Journal of Computational Physics},
        volume={274},
        pages={826--840},
        year={2014},
        publisher={Elsevier}
    }
"""
import firedrake as fe
import sapphire.simulations.enthalpy_porosity
import typing


class Simulation(sapphire.simulations.enthalpy_porosity.Simulation):
    
    def __init__(self, *args, 
            mesh_dimensions = (24, 24),
            hotwall_temperature = 1.,
            initial_temperature = -0.01,
            reynolds_number = 1.,
            rayleigh_number = 3.27e5,
            prandtl_number = 56.2,
            stefan_number = 0.045,
            liquidus_temperature = 0.,
            **kwargs):
        
        if "solution" not in kwargs:
            
            kwargs["mesh"] = fe.UnitSquareMesh(*mesh_dimensions)
            
        self.hotwall_temperature = fe.Constant(hotwall_temperature)
        
        self.initial_temperature = fe.Constant(initial_temperature)
        
        super().__init__(
            *args,
            liquidus_temperature = liquidus_temperature,
            reynolds_number = reynolds_number,
            rayleigh_number = rayleigh_number,
            prandtl_number = prandtl_number,
            stefan_number = stefan_number,
            **kwargs)
    
    def initial_values(self):
        
        _, _, T = self.solution.split()
        
        T.assign(self.initial_temperature)
        
        return self.solution
        
    def dirichlet_boundary_conditions(self):
    
        W = self.solution_space
        
        return [
            fe.DirichletBC(
                W.sub(1), (0.,)*self.mesh.geometric_dimension(), "on_boundary"),
            fe.DirichletBC(W.sub(2), self.hotwall_temperature, 1),
            fe.DirichletBC(W.sub(2), self.initial_temperature, 2)]
            