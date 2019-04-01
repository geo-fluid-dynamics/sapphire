""" A steady incompressible Navier-Stokes-Boussinesq simulation class """
import firedrake as fe
import sunfire.simulation


inner, dot, grad, div, sym = \
        fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
def variational_form_residual(sim, solution):
    
    Gr = sim.grashof_number
    
    Pr = sim.prandtl_number
    
    ihat, jhat = sunfire.simulation.unit_vectors(sim.mesh)
    
    sim.gravity_direction = fe.Constant(-jhat)
    
    ghat = sim.gravity_direction
    
    p, u, T = fe.split(solution)
    
    psi_p, psi_u, psi_T = fe.TestFunctions(solution.function_space())
    
    mass = psi_p*div(u)
    
    momentum = dot(psi_u, grad(u)*u + Gr*T*ghat) \
        - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), sym(grad(u)))
    
    energy = psi_T*dot(u, grad(T)) + dot(grad(psi_T), 1./Pr*grad(T))
    
    dx = fe.dx(degree = sim.quadrature_degree)
    
    return (mass + momentum + energy)*dx
    
    
def strong_residual(sim, solution):
    
    Gr = sim.grashof_number
    
    Pr = sim.prandtl_number
    
    ghat = sim.gravity_direction
    
    p, u, T = solution
    
    r_p = div(u)
    
    r_u = grad(u)*u + grad(p) - 2.*div(sym(grad(u))) + Gr*T*ghat
    
    r_T = dot(u, grad(T)) - 1./Pr*div(grad(T))
    
    return r_p, r_u, r_T
    
    
def element(cell, degree):
    
    scalar = fe.FiniteElement("P", cell, degree)
    
    vector = fe.VectorElement("P", cell, degree + 1)
    
    return fe.MixedElement(scalar, vector, scalar)
    
    
class Simulation(sunfire.simulation.Simulation):
    
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
            