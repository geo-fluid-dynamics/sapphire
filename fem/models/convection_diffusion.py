""" A convection-diffusion model class """
import firedrake as fe
import fem.model

    
class Model(fem.model.Model):
    
    def __init__(self):
    
        self.kinematic_viscosity = fe.Constant(1.)
    
        super().__init__()
        
    def init_element(self):
    
        self.element = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
    
    def init_weak_form_residual(self):
        
        u, v = self.solution, fe.TestFunction(self.solution.function_space())
        
        x = fe.SpatialCoordinate(self.solution.function_space().mesh())
        
        a = self.advection_velocity
        
        nu = self.kinematic_viscosity
        
        dot, grad = fe.dot, fe.grad
        
        self.weak_form_residual = v*dot(a, grad(u)) + dot(grad(v), nu*grad(u))
        