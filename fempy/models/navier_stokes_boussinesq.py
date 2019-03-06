""" A steady incompressible Navier-Stokes-Boussinesq model class """
import firedrake as fe
import fempy.model


inner, dot, grad, div, sym = \
        fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
def variational_form_residual(model, solution):
    
    Gr = model.grashof_number
    
    Pr = model.prandtl_number
    
    ihat, jhat = fempy.model.unit_vectors(model.mesh)
    
    model.gravity_direction = fe.Constant(-jhat)
    
    ghat = model.gravity_direction
    
    p, u, T = fe.split(solution)
    
    psi_p, psi_u, psi_T = fe.TestFunctions(solution.function_space())
    
    mass = psi_p*div(u)
    
    momentum = dot(psi_u, grad(u)*u + Gr*T*ghat) \
        - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), sym(grad(u)))
    
    energy = psi_T*dot(u, grad(T)) + dot(grad(psi_T), 1./Pr*grad(T))
    
    dx = fe.dx(degree = model.quadrature_degree)
    
    return (mass + momentum + energy)*dx
    
    
def strong_residual(model, solution):
    
    Gr = model.grashof_number
    
    Pr = model.prandtl_number
    
    ghat = model.gravity_direction
    
    p, u, T = solution
    
    r_p = div(u)
    
    r_u = grad(u)*u + grad(p) - 2.*div(sym(grad(u))) + Gr*T*ghat
    
    r_T = dot(u, grad(T)) - 1./Pr*div(grad(T))
    
    return r_p, r_u, r_T
    
    
def element(cell, degree):
    
    scalar = fe.FiniteElement("P", cell, degree)
    
    vector = fe.VectorElement("P", cell, degree + 1)
    
    return fe.MixedElement(scalar, vector, scalar)
    
    
class Model(fempy.model.Model):
    
    def __init__(self, *args, mesh, element_degree, **kwargs):
        
        self.grashof_number = fe.Constant(1.)
        
        self.prandtl_number = fe.Constant(1.)
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            variational_form_residual = variational_form_residual,
            time_dependent = False,
            **kwargs)
            