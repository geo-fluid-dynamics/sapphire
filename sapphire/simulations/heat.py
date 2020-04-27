""" A heat simulation class """
import firedrake as fe
import sapphire.simulation

    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args, 
            element_degree = 1, 
            solver_parameters = {"ksp_type": "cg"},
            **kwargs):
        
        if "solution" not in kwargs:
            
            mesh = kwargs["mesh"]
            
            del kwargs["mesh"]
            
            element = fe.FiniteElement("P", mesh.ufl_cell(), element_degree)
            
            kwargs["solution"] = fe.Function(fe.FunctionSpace(mesh, element))
            
        super().__init__(*args,
            solver_parameters = solver_parameters, **kwargs)
    
    def weak_form_residual(self):
        
        u = self.solution
        
        u_t = super().time_discrete_terms()
        
        v = fe.TestFunction(self.solution_space)
        
        dot, grad = fe.dot, fe.grad
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return (v*u_t + dot(grad(v), grad(u)))*dx


def strong_residual(sim, solution):
    
    u = solution
    
    t = sim.time
    
    diff, div, grad = fe.diff, fe.div, fe.grad
    
    return diff(u, t) - div(grad(u))
    