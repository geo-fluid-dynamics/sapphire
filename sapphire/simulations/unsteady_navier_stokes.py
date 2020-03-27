"""Provides a simulation class governed by Navier-Stokes. 

This can be used to simulate incompressible flow,
e.g. the lid-driven cavity.
"""
import firedrake as fe
import sapphire.simulation


diff, inner, dot, grad, div, sym = \
    fe.diff, fe.inner, fe.dot, fe.grad, fe.div, fe.sym
    
def weak_form_residual(sim, solution):
    
    u, p = fe.split(sim.solution)
    
    u_t, _ = sim.time_discrete_terms()
    
    Re = sim.reynolds_number
    
    psi_u, psi_p = fe.TestFunctions(sim.solution.function_space())
    
    mass = psi_p*div(u)
    
    momentum = dot(psi_u, u_t + grad(u)*u) - div(psi_u)*p + \
        2./Re*inner(sym(grad(psi_u)), sym(grad(u)))
    
    dx = fe.dx(degree = sim.quadrature_degree)
    
    return (mass + momentum)*dx
    
    
def strong_residual(sim, solution):
    
    u, p = solution
    
    t = sim.time
    
    Re = sim.reynolds_number
    
    r_u = diff(u, t) + grad(u)*u + grad(p) - 2./Re*div(sym(grad(u)))
    
    r_p = div(u)
    
    return r_u, r_p
    
    
def element(cell, degree):

    vector = fe.VectorElement("P", cell, degree[0])
    
    scalar = fe.FiniteElement("P", cell, degree[1])
    
    return fe.MixedElement(vector, scalar)
    
    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args,
            mesh,
            element_degree,
            reynolds_number,
            **kwargs):
            
        self.reynolds_number = fe.Constant(reynolds_number)
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            weak_form_residual = weak_form_residual,
            **kwargs)
            