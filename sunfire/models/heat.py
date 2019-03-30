""" A heat model class """
import firedrake as fe
import sunfire.model

    
def variational_form_residual(model, solution):
    
    u = solution
    
    u_t = sunfire.model.time_discrete_terms(
        solutions = model.solutions, timestep_size = model.timestep_size)
    
    v = fe.TestFunction(solution.function_space())
    
    dot, grad = fe.dot, fe.grad
    
    dx = fe.dx(degree = model.quadrature_degree)
    
    return (v*u_t + dot(grad(v), grad(u)))*dx
    
    
def element(cell, degree):

    return fe.FiniteElement("P", cell, degree)
    
    
def strong_residual(model, solution):
        
        u = solution
        
        t = model.time
        
        diff, div, grad = fe.diff, fe.div, fe.grad
        
        return diff(u, t) - div(grad(u))
        
    
class Model(sunfire.model.Model):
    
    def __init__(self, *args, mesh, element_degree, **kwargs):
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            variational_form_residual = variational_form_residual,
            **kwargs)
    
    def solve(self, *args, **kwargs):
        
        return super().solve(*args, parameters = {"ksp_type": "cg"}, **kwargs)
        