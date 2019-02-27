""" A heat model class """
import firedrake as fe
import fempy.unsteady_model

    
class Model(fempy.unsteady_model.Model):
    
    def init_element(self):
    
        self.element = fe.FiniteElement(
            "P", self.mesh.ufl_cell(), self.spatial_order - 1)
    
    def init_weak_form_residual(self):
        
        u = self.solution
        
        u_t = self.time_discrete_terms
        
        v = fe.TestFunction(self.function_space)
        
        dot, grad = fe.dot, fe.grad
        
        self.weak_form_residual = v*u_t + dot(grad(v), grad(u))
    