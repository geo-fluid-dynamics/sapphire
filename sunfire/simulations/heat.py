""" A heat simulation class """
import firedrake as fe
import sunfire.simulation

    
def variational_form_residual(sim, solution):
    
    u = solution
    
    u_t = sunfire.simulation.time_discrete_terms(
        solutions = sim.solutions, timestep_size = sim.timestep_size)
    
    v = fe.TestFunction(solution.function_space())
    
    dot, grad = fe.dot, fe.grad
    
    dx = fe.dx(degree = sim.quadrature_degree)
    
    return (v*u_t + dot(grad(v), grad(u)))*dx
    
    
def element(cell, degree):

    return fe.FiniteElement("P", cell, degree)
    
    
def strong_residual(sim, solution):
        
        u = solution
        
        t = sim.time
        
        diff, div, grad = fe.diff, fe.div, fe.grad
        
        return diff(u, t) - div(grad(u))
        
    
class Simulation(sunfire.simulation.Simulation):
    
    def __init__(self, *args, mesh, element_degree, **kwargs):
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            variational_form_residual = variational_form_residual,
            **kwargs)
    
    def solve(self, *args, **kwargs):
        
        return super().solve(*args, parameters = {"ksp_type": "cg"}, **kwargs)
        