"""Provides a simulation class governed by Navier-Stokes. 

This can be used to simulate incompressible flow,
e.g. the lid-driven cavity.

Dirichlet BC's should not be placed on the pressure.
The returned pressure solution will always have zero mean.

Non-homogeneous Neumann BC's are not implemented for the velocity.
"""
import firedrake as fe
import sapphire.simulation


def element(cell, degree):
    
    return fe.MixedElement(
        fe.VectorElement("P", cell, degree[0]),
        fe.FiniteElement("P", cell, degree[1]))
    
    
inner, dot, grad, div, sym = \
    fe.inner, fe.dot, fe.grad, fe.div, fe.sym
    
diff = fe.diff
    
def weak_form_residual(sim, solution):
    
    u, p = fe.split(solution)
    
    u_t, _ = sim.time_discrete_terms()
    
    Re = sim.reynolds_number
    
    psi_u, psi_p = fe.TestFunctions(solution.function_space())
    
    mass = psi_p*div(u)
    
    momentum = dot(psi_u, u_t + grad(u)*u) - div(psi_u)*p + \
        2./Re*inner(sym(grad(psi_u)), sym(grad(u)))
    
    gamma = sim.pressure_penalty_constant
    
    pressure_penalty = gamma*psi_p*p
    
    dx = fe.dx(degree = sim.quadrature_degree)
    
    return (mass + momentum + pressure_penalty)*dx
    
    
def strong_residual(sim, solution):
    
    u, p = solution
    
    t = sim.time
    
    Re = sim.reynolds_number
    
    r_u = diff(u, t) + grad(u)*u + grad(p) - 2./Re*div(sym(grad(u)))
    
    r_p = div(u)
    
    return r_u, r_p


def nullspace(sim):
    """Inform solver that pressure is only defined up to a constant."""
    W = sim.function_space
    
    return fe.MixedVectorSpaceBasis(
        W, [W.sub(0), fe.VectorSpaceBasis(constant=True)])
        
    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args,
            mesh,
            reynolds_number,
            pressure_penalty_constant = 0.,
            element_degree = (2, 1),
            **kwargs):
            
        self.reynolds_number = fe.Constant(reynolds_number)
        
        self.pressure_penalty_constant = fe.Constant(
            pressure_penalty_constant)
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            weak_form_residual = weak_form_residual,
            nullspace = nullspace,
            **kwargs)
            
    def solve(self) -> fe.Function:
        
        self.solution = super().solve()
        
        u, p = self.solution.split()
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        mean_pressure = fe.assemble(p*dx)
        
        p = p.assign(p - mean_pressure)
        
        return self.solution
        