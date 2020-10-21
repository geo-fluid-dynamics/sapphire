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
            solver_parameters = solver_parameters,
            fieldnames = ('T',),
            **kwargs)
    
    def weak_form_residual(self):
        
        T = self.solution
        
        T_t = self.time_discrete_terms['T']
        
        psi = fe.TestFunction(self.solution_space)
        
        dot, grad = fe.dot, fe.grad
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return (psi*T_t + dot(grad(psi), grad(T)))*dx
