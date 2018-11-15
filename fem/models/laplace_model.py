""" **laplace_model.py** 
implements a Laplace model class. 
"""
import firedrake as fe
import fem.abstract_model

    
class LaplaceModel(fem.abstract_model.AbstractModel):
    
    def set_element(self):
    
        self.element = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
    
    def weak_form_residual(self):
        
        u, v = self.solution, fe.TestFunction(self.solution.function_space())
        
        dot, grad = fe.dot, fe.grad
        
        return - dot(grad(v), grad(u))
    