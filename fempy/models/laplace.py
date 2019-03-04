""" A Laplace model class """
import firedrake as fe
import fempy.model

    
class Model(fempy.model.Model):
    
    def __init__(self, *args, mesh, element_degree, **kwargs):
    
        element = fe.FiniteElement("P", mesh.ufl_cell(), element_degree)
        
        super().__init__(*args, mesh, element, **kwargs)
    
    def init_weak_form_residual(self):
        
        u = self.solution
        
        v = fe.TestFunction(self.function_space)
        
        dot, grad = fe.dot, fe.grad
        
        self.weak_form_residual = - dot(grad(v), grad(u))

    def strong_form_residual(self, solution):
    
        div, grad, = fe.div, fe.grad
        
        u = solution
        
        return div(grad(u))
        