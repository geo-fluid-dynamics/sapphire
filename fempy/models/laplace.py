""" A Laplace model class "
"""
import firedrake as fe
import fempy.model

    
class Model(fempy.model.Model):
    
    def init_element(self):
    
        self.element = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
    
    def init_weak_form_residual(self):
        
        u = self.solution
        
        v = fe.TestFunction(self.function_space)
        
        dot, grad = fe.dot, fe.grad
        
        self.weak_form_residual = - dot(grad(v), grad(u))
