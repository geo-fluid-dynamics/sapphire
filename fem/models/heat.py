""" A heat model class """
import firedrake as fe
import fem.unsteady_model

    
class Model(fem.unsteady_model.UnsteadyModel):
    
    def __init__(self):
        
        self.thermal_diffusivity = fe.Constant(1.)
        
        super().__init__()
        
    def init_element(self):
    
        self.element = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
    
    def init_weak_form_residual(self):
        
        alpha = self.thermal_diffusivity
        
        u = self.solution
        
        un = self.initial_values
        
        Delta_t = self.timestep_size
        
        u_t = (u - un)/Delta_t
        
        v = fe.TestFunction(self.function_space)
        
        dot, grad = fe.dot, fe.grad
        
        self.weak_form_residual = v*u_t + alpha*dot(grad(v), grad(u))
    