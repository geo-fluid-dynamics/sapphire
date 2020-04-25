"""Provides a simulation class governed by Navier-Stokes-Boussinesq.

This can be used to simulate natural convection,
e.g the heat-driven cavity.

Dirichlet BC's should not be placed on the pressure.
The returned pressure solution will always have zero mean.

Non-homogeneous Neumann BC's are not implemented for the velocity.
"""
import firedrake as fe
import sapphire.simulation

    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args, 
            mesh, 
            element_degree = (1, 2, 2),
            grashof_number = 1.,
            prandtl_number = 1.,
            buoyancy = None,
            **kwargs):
        
        if buoyancy is None:
        
            buoyancy = linear_boussinesq_buoyancy
            
        self.buoyancy = buoyancy
        
        self.grashof_number = fe.Constant(grashof_number)
        
        self.prandtl_number = fe.Constant(prandtl_number)
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            weak_form_residual = weak_form_residual,
            nullspace = nullspace,
            **kwargs)
            
    def solve(self) -> fe.Function:
        
        self.solution = super().solve()
        
        p, u, T = self.solution.split()
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        mean_pressure = fe.assemble(p*dx)
        
        p = p.assign(p - mean_pressure)
        
        return self.solution


diff, inner, dot, grad, div, sym = \
    fe.diff, fe.inner, fe.dot, fe.grad, fe.div, fe.sym
    
def weak_form_residual(sim, solution):
    
    Pr = sim.prandtl_number
    
    p, u, T = fe.split(solution)
    
    _, u_t, T_t = sim.time_discrete_terms()
    
    psi_p, psi_u, psi_T = fe.TestFunctions(solution.function_space())
    
    b = sim.buoyancy(sim = sim, temperature = T)
    
    mass = psi_p*div(u)
    
    momentum = dot(psi_u, u_t + grad(u)*u + b) \
        - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), sym(grad(u)))
    
    energy = psi_T*(T_t + dot(u, grad(T))) + dot(grad(psi_T), 1./Pr*grad(T))
    
    dx = fe.dx(degree = sim.quadrature_degree)
    
    return (mass + momentum + energy)*dx
    
    
def element(cell, degree):
    
    pdeg, udeg, Tdeg = degree
    
    pressure_element = fe.FiniteElement("P", cell, pdeg)
    
    velocity_element = fe.VectorElement("P", cell, udeg)
    
    temperature_element = fe.FiniteElement("P", cell, Tdeg)
    
    return fe.MixedElement(
        pressure_element, velocity_element, temperature_element)


def nullspace(sim):
    """Inform solver that pressure solution is not unique.
    
    It is only defined up to adding an arbitrary constant.
    """
    W = sim.function_space
    
    return fe.MixedVectorSpaceBasis(
        W, [fe.VectorSpaceBasis(constant=True), W.sub(1), W.sub(2)])


def linear_boussinesq_buoyancy(sim, temperature):
    
    T = temperature
    
    Gr = sim.grashof_number
    
    ghat = fe.Constant(-sim.unit_vectors()[1])
    
    return Gr*T*ghat
    
    
def strong_residual(sim, solution):
    
    Pr = sim.prandtl_number
    
    p, u, T = solution
    
    t = sim.time
    
    b = sim.buoyancy(sim = sim, temperature = T)
    
    r_p = div(u)
    
    r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(sym(grad(u))) + b
    
    r_T = diff(T, t) + dot(u, grad(T)) - 1./Pr*div(grad(T))
    
    return r_p, r_u, r_T
    