""" **heat_model.py** 
implements a heat model class. 
"""
import firedrake as fe
import fem.abstract_unsteady_model

    
class HeatModel(fem.abstract_unsteady_model.AbstractUnsteadyModel):
    
    def element(self):
    
        return fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
    
    def weak_form_residual(self):
        
        alpha = self.residual_parameters["thermal_diffusivity"]
        
        u = self.solution
        
        un = self.initial_values
        
        Delta_t = self.timestep_size
        
        u_t = (u - un)/Delta_t
        
        v = fe.TestFunction(self.solution.function_space())
        
        dot, grad = fe.dot, fe.grad
        
        return u_t + alpha*dot(grad(v), grad(u))
    