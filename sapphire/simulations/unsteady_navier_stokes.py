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
    
    def momentum(self):
        
        u_t = self.time_discrete_terms["u"]
        
        psi_u = self.test_functions["u"]
        
        return super().momentum() + dot(psi_u, u_t)*self.dx
    