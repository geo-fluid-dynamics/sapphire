""" A convection-diffusion model class """
import firedrake as fe
import fempy.model


dot, grad, div = fe.dot, fe.grad, fe.div
    
def variational_form_residual(model, solution):
    
    u = solution
    
    v = fe.TestFunction(solution.function_space())
    
    x = fe.SpatialCoordinate(solution.function_space().mesh())
    
    a = model.advection_velocity
    
    nu = model.kinematic_viscosity
    
    return v*dot(a, grad(u)) + dot(grad(v), nu*grad(u))
    
    
def strong_form_residual(model, solution):
    
    x = fe.SpatialCoordinate(model.mesh)
    
    u = solution
    
    a = model.advection_velocity
    
    nu = model.kinematic_viscosity
    
    return dot(a, grad(u)) - div(nu*grad(u))


def element(cell, degree):

    return fe.FiniteElement("P", cell, degree)
    
    
class Model(fempy.model.Model):
    
    def __init__(self, 
            *args, 
            mesh,
            element_degree,
            advection_velocity,
            **kwargs):
        
        self.kinematic_viscosity = fe.Constant(1.)
        
        self.advection_velocity = advection_velocity(mesh)
    
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            variational_form_residual = variational_form_residual,
            **kwargs)
        