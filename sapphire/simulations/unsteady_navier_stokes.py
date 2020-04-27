"""Provides a simulation class governed by Navier-Stokes. 

This can be used to simulate incompressible flow,
e.g. the lid-driven cavity.

Dirichlet BC's should not be placed on the pressure.
The returned pressure solution will always have zero mean.

Non-homogeneous Neumann BC's are not implemented.
"""
import firedrake as fe
import sapphire.simulations.navier_stokes


inner, dot, grad, div, sym = \
    fe.inner, fe.dot, fe.grad, fe.div, fe.sym
    
class Simulation(sapphire.simulations.navier_stokes.Simulation):
    
    def __init__(self, *args, **kwargs):
        
        if "time_stencil_size" not in kwargs:
        
            kwargs["time_stencil_size"] = 2
            
        super().__init__(*args, **kwargs)
    
    def momentum(self):
        
        u_t = self.time_discrete_terms()
        
        psi_u, _ = fe.TestFunctions(self.solution_space)
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return super().momentum() + dot(psi_u, u_t)*dx
        
    def time_discrete_terms(self):
        
        u_t, _ = sapphire.Simulation.time_discrete_terms(self)
        
        return u_t


diff = fe.diff

def strong_residual(sim, solution):
    
    u, p = solution
    
    t = sim.time
    
    Re = sim.reynolds_number
    
    r_u = diff(u, t) + grad(u)*u + grad(p) - 2./Re*div(sym(grad(u)))
    
    r_p = div(u)
    
    return r_u, r_p
    