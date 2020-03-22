"""Provides a simulation class governed by Navier-Stokes-Boussinesq.

This can be used to simulate natural convection,
e.g the heat-driven cavity.
"""
import firedrake as fe
import sapphire.simulation


def linear_boussinesq_buoyancy(sim, temperature):
    
    T = temperature
    
    Gr = sim.grashof_number
    
    ghat = fe.Constant(-sim.unit_vectors()[1])
    
    return Gr*T*ghat
    
    
_,       diff,    inner,    dot,    grad,    div,    sym = \
None, fe.diff, fe.inner, fe.dot, fe.grad, fe.div, fe.sym
    
def weak_form_residual(
        sim, solution, buoyancy = linear_boussinesq_buoyancy):
    
    Pr = sim.prandtl_number
    
    p, u, T = fe.split(solution)
    
    _, u_t, T_t = sim.time_discrete_terms()
    
    psi_p, psi_u, psi_T = fe.TestFunctions(solution.function_space())
    
    b = buoyancy(sim = sim, temperature = T)
    
    mass = psi_p*div(u)
    
    momentum = dot(psi_u, u_t + grad(u)*u + b) \
        - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), sym(grad(u)))
    
    energy = psi_T*(T_t + dot(u, grad(T))) + dot(grad(psi_T), 1./Pr*grad(T))
    
    gamma = sim.pressure_penalty_factor
    
    pressure_penalty = gamma*psi_p*p
    
    dx = fe.dx(degree = sim.quadrature_degree)
    
    return (mass + momentum + energy + pressure_penalty)*dx
    
    
def strong_residual(sim, solution, buoyancy = linear_boussinesq_buoyancy):
    
    Pr = sim.prandtl_number
    
    p, u, T = solution
    
    t = sim.time
    
    b = buoyancy(sim = sim, temperature = T)
    
    r_p = div(u)
    
    r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(sym(grad(u))) + b
    
    r_T = diff(T, t) + dot(u, grad(T)) - 1./Pr*div(grad(T))
    
    return r_p, r_u, r_T
    
    
def element(cell, degree):
    
    pdeg, udeg, Tdeg = degree
    
    pressure_element = fe.FiniteElement("P", cell, pdeg)
    
    velocity_element = fe.VectorElement("P", cell, udeg)
    
    temperature_element = fe.FiniteElement("P", cell, Tdeg)
    
    return fe.MixedElement(
        pressure_element, velocity_element, temperature_element)

    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args, 
            mesh, 
            element_degree = (1, 2, 2),
            grashof_number = 1.,
            prandtl_number = 1.,
            pressure_penalty_factor = 0.,
            **kwargs):
        
        self.grashof_number = fe.Constant(grashof_number)
        
        self.prandtl_number = fe.Constant(prandtl_number)
        
        self.pressure_penalty_factor = fe.Constant(pressure_penalty_factor)
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            weak_form_residual = weak_form_residual,
            **kwargs)
            