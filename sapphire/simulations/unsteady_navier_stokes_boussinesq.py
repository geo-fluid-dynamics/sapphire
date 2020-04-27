"""Provides a simulation class governed by Navier-Stokes-Boussinesq.

This can be used to simulate natural convection,
e.g the heat-driven cavity.

Dirichlet BC's should not be placed on the pressure.
The returned pressure solution will always have zero mean.

Non-homogeneous Neumann BC's are not implemented.
"""
import firedrake as fe
import sapphire.simulations.navier_stokes_boussinesq


inner, dot, grad, div, sym = \
    fe.inner, fe.dot, fe.grad, fe.div, fe.sym

class Simulation(sapphire.simulations.navier_stokes_boussinesq.Simulation):
    
    def momentum(self):
        
        u_t, _ = self.time_discrete_terms()
        
        _, psi_u, _ = fe.TestFunctions(self.solution_space)
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return super().momentum() + dot(psi_u, u_t)*dx
    
    def energy(self):
        
        _, T_t = self.time_discrete_terms()
        
        _, _, psi_T = fe.TestFunctions(self.solution_space)
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return super().energy() + psi_T*T_t*dx
    
    def time_discrete_terms(self):
    
        _, u_t, T_t = sapphire.Simulation.time_discrete_terms(self)
        
        return u_t, T_t


diff = fe.diff

def strong_residual(sim, solution):
    
    Pr = sim.prandtl_number
    
    p, u, T = solution
    
    t = sim.time
    
    b = sim.buoyancy(temperature = T)
    
    r_p = div(u)
    
    r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(sym(grad(u))) + b
    
    r_T = diff(T, t) + dot(u, grad(T)) - 1./Pr*div(grad(T))
    
    return r_p, r_u, r_T
    