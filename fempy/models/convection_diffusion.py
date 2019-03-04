""" A convection-diffusion model class """
import firedrake as fe
import fempy.model

    
class Model(fempy.model.Model):
    
    def __init__(self, 
            *args, 
            mesh, element_degree, advection_velocity,
            **kwargs):
        
        self.kinematic_viscosity = fe.Constant(1.)
        
        self.advection_velocity = advection_velocity(mesh)
    
        element = fe.FiniteElement("P", mesh.ufl_cell(), element_degree)
        
        super().__init__(*args, mesh = mesh, element = element, **kwargs)
        
    def init_weak_form_residual(self):
        
        u, v = self.solution, fe.TestFunction(self.solution.function_space())
        
        x = fe.SpatialCoordinate(self.solution.function_space().mesh())
        
        a = self.advection_velocity
        
        nu = self.kinematic_viscosity
        
        dot, grad = fe.dot, fe.grad
        
        self.weak_form_residual = v*dot(a, grad(u)) + dot(grad(v), nu*grad(u))
        
    def strong_form_residual(self, solution):
        
        x = fe.SpatialCoordinate(self.mesh)
        
        u = solution
        
        a = self.advection_velocity
        
        nu = self.kinematic_viscosity
        
        dot, grad, div = fe.dot, fe.grad, fe.div
        
        return dot(a, grad(u)) - div(nu*grad(u))
        