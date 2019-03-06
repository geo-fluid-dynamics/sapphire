""" A Laplace model class """
import firedrake as fe
import fempy.model

    
def variational_form_residual(model, solution):
    
    u = solution
    
    v = fe.TestFunction(solution.function_space())
    
    dot, grad = fe.dot, fe.grad
    
    dx = fe.dx(degree = model.quadrature_degree)
    
    return -dot(grad(v), grad(u))*dx
    

def element(cell, degree):
    
    return fe.FiniteElement("P", cell, degree)
    
    
def strong_residual(model, solution):
    
    div, grad, = fe.div, fe.grad
    
    u = solution
    
    return div(grad(u))
    
    
class Model(fempy.model.Model):
    
    def __init__(self, *args, mesh, element_degree, **kwargs):
    
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            variational_form_residual = variational_form_residual,
            time_dependent = False,
            **kwargs)
    
    def solve(self, *args, **kwargs):
        
        return super().solve(*args, parameters = {"ksp_type": "cg"}, **kwargs)
    