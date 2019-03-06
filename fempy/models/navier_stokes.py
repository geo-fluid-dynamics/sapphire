""" A steady incompressible Navier-Stokes model class """
import firedrake as fe
import fempy.model


inner, dot, grad, div, sym = \
        fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
def variational_form_residual(model, solution):
    
    u, p = fe.split(solution)
    
    psi_u, psi_p = fe.TestFunctions(solution.function_space())
    
    mass = psi_p*div(u)
    
    momentum = dot(psi_u, grad(u)*u) - div(psi_u)*p + \
        2.*inner(sym(grad(psi_u)), sym(grad(u)))
    
    dx = fe.dx(degree = model.quadrature_degree)
    
    return (mass + momentum)*dx
    
    
def element(cell, degree):

    return fe.MixedElement(
        fe.VectorElement("P", cell, degree + 1),
        fe.FiniteElement("P", cell, degree))
        
        
def strong_residual(model, solution):
    
    u, p = solution
    
    r_u = grad(u)*u + grad(p) - 2.*div(sym(grad(u)))
    
    r_p = div(u)
    
    return r_u, r_p
    
    
class Model(fempy.model.Model):
    
    def __init__(self, *args, mesh, element_degree, **kwargs):
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            variational_form_residual = variational_form_residual,
            time_dependent = False,
            **kwargs)
        