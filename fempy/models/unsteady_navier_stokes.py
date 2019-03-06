""" Unsteady incompressible Navier-Stokes model """
import firedrake as fe
import fempy.model


diff, inner, dot, grad, div, sym = \
    fe.diff, fe.inner, fe.dot, fe.grad, fe.div, fe.sym
    
def variational_form_residual(model, solution):
    
    u, p = fe.split(model.solution)
    
    u_t, _ = fempy.model.time_discrete_terms(
        solutions = model.solutions, timestep_size = model.timestep_size)
    
    psi_u, psi_p = fe.TestFunctions(model.solution.function_space())
    
    mass = psi_p*div(u)
    
    momentum = dot(psi_u, u_t + grad(u)*u) - div(psi_u)*p + \
        2.*inner(sym(grad(psi_u)), sym(grad(u)))
    
    dx = fe.dx(degree = model.quadrature_degree)
    
    return (mass + momentum)*dx
    
    
def strong_residual(model, solution):
    
    u, p = solution
    
    t = model.time
    
    r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(sym(grad(u)))
    
    r_p = div(u)
    
    return r_u, r_p
    
    
def element(cell, degree):

    vector = fe.VectorElement("P", cell, degree + 1)
    
    scalar = fe.FiniteElement("P", cell, degree)
    
    return fe.MixedElement(vector, scalar)
    
    
class Model(fempy.model.Model):
    
    def __init__(self, *args, mesh, element_degree, **kwargs):
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            variational_form_residual = variational_form_residual,
            **kwargs)
            