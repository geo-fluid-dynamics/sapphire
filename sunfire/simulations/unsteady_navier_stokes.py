""" Unsteady incompressible Navier-Stokes simulation """
import firedrake as fe
import sunfire.simulation


diff, inner, dot, grad, div, sym = \
    fe.diff, fe.inner, fe.dot, fe.grad, fe.div, fe.sym
    
def variational_form_residual(sim, solution):
    
    u, p = fe.split(sim.solution)
    
    u_t, _ = sunfire.simulation.time_discrete_terms(
        solutions = sim.solutions, timestep_size = sim.timestep_size)
    
    psi_u, psi_p = fe.TestFunctions(sim.solution.function_space())
    
    mass = psi_p*div(u)
    
    momentum = dot(psi_u, u_t + grad(u)*u) - div(psi_u)*p + \
        2.*inner(sym(grad(psi_u)), sym(grad(u)))
    
    dx = fe.dx(degree = sim.quadrature_degree)
    
    return (mass + momentum)*dx
    
    
def strong_residual(sim, solution):
    
    u, p = solution
    
    t = sim.time
    
    r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(sym(grad(u)))
    
    r_p = div(u)
    
    return r_u, r_p
    
    
def element(cell, degree):

    vector = fe.VectorElement("P", cell, degree + 1)
    
    scalar = fe.FiniteElement("P", cell, degree)
    
    return fe.MixedElement(vector, scalar)
    
    
class Simulation(sunfire.simulation.Simulation):
    
    def __init__(self, *args, mesh, element_degree, **kwargs):
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            variational_form_residual = variational_form_residual,
            **kwargs)
            