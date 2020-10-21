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
        
        u_t = self.time_discrete_terms["u"]
        
        psi_u = self.test_functions["u"]
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return super().momentum() + dot(psi_u, u_t)*dx
    
    def energy(self):
        
        T_t = self.time_discrete_terms["T"]
        
        psi_T = self.test_functions["T"]
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return super().energy() + psi_T*T_t*dx
        