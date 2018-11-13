""" **convection_diffusion_model.py** 
implements a convection-diffusion model class. 
"""
import firedrake as fe
import fem.abstract_model

    
class ConvectionDiffusionModel(fem.abstract_model.AbstractModel):
    
    def element(self):
    
        return fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
    
    def weak_form_residual(self):
        
        u, v = self.solution, fe.TestFunction(self.solution.function_space())
        
        x = fe.SpatialCoordinate(self.solution.function_space().mesh())
        
        a = self.residual_parameters["velocity"](x)
        
        nu = self.residual_parameters["viscosity"]
        
        dot, grad = fe.dot, fe.grad
        
        return v*dot(a, grad(u)) + dot(grad(v), nu*grad(u))
        