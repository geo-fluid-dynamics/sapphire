""" A convection-diffusion model class """
import firedrake as fe
import fem.model

    
class Model(fem.model.Model):
    
    def __init__(self):
    
        super().__init__()
        
        self.kinematic_viscosity = fe.Constant(1.)
        
    def init_element(self):
    
        self.element = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
    
    def weak_form_residual(self):
        
        u, v = self.solution, fe.TestFunction(self.solution.function_space())
        
        x = fe.SpatialCoordinate(self.solution.function_space().mesh())
        
        a = self.advection_velocity
        
        nu = self.kinematic_viscosity
        
        dot, grad = fe.dot, fe.grad
        
        return v*dot(a, grad(u)) + dot(grad(v), nu*grad(u))
        