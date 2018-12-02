""" A heat model class """
import firedrake as fe
import fempy.unsteady_model

    
class Model(fempy.unsteady_model.Model):
    
    def __init__(self):
        
        super().__init__()
        
    def init_element(self):
    
        self.element = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
    
    def init_time_discrete_terms(self):
        """ Implicit Euler finite difference scheme """
        u = self.solution
        
        un = self.initial_values
        
        Delta_t = self.timestep_size
        
        self.time_discrete_terms = (u - un)/Delta_t
    
    def init_weak_form_residual(self):
        
        u = self.solution
        
        u_t = self.time_discrete_terms
        
        v = fe.TestFunction(self.function_space)
        
        dot, grad = fe.dot, fe.grad
        
        self.weak_form_residual = v*u_t + dot(grad(v), grad(u))
    