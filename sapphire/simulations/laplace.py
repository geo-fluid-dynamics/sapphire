""" A Laplace simulation class """
import firedrake as fe
import sapphire.simulation

    
def variational_form_residual(sim, solution):
    
    u = solution
    
    v = fe.TestFunction(solution.function_space())
    
    dot, grad = fe.dot, fe.grad
    
    dx = fe.dx(degree = sim.quadrature_degree)
    
    return -dot(grad(v), grad(u))*dx
    

def element(cell, degree):
    
    return fe.FiniteElement("P", cell, degree)
    
    
def strong_residual(sim, solution):
    
    div, grad, = fe.div, fe.grad
    
    u = solution
    
    return div(grad(u))
    
    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args, 
            mesh, 
            element_degree = 1, 
            solver_parameters = {"ksp_type": "cg"},
            **kwargs):
    
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            variational_form_residual = variational_form_residual,
            time_dependent = False,
            solver_parameters = solver_parameters,
            **kwargs)
    