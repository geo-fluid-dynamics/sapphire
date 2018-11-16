""" A Laplace model class "
"""
import firedrake as fe
import fem.model

    
class Model(fem.model.Model):
    
    def init_element(self):
    
        self.element = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
    
    def weak_form_residual(self):
        
        u = self.solution
        
        v = fe.TestFunction(self.function_space)
        
        dot, grad = fe.dot, fe.grad
        
        return - dot(grad(v), grad(u))
    