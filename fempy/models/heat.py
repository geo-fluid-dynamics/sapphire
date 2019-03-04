""" A heat model class """
import firedrake as fe
import fempy.unsteady_model

    
class Model(fempy.unsteady_model.UnsteadyModel):
    
    def __init__(self, *args, mesh, element_degree, **kwargs):
    
        element = fe.FiniteElement("P", mesh.ufl_cell(), element_degree)
        
        super().__init__(*args, mesh, element, **kwargs)
    
    def init_weak_form_residual(self):
        
        u = self.solution
        
        u_t = self.time_discrete_terms
        
        v = fe.TestFunction(self.function_space)
        
        dot, grad = fe.dot, fe.grad
        
        self.weak_form_residual = v*u_t + dot(grad(v), grad(u))
    
    def strong_form_residual(self, solution):
        
        u = solution
        
        t = self.time
        
        diff, div, grad = fe.diff, fe.div, fe.grad
        
        return diff(u, t) - div(grad(u))
        
    def init_solver(self, solver_parameters = {"ksp_type": "cg"}):
        
        self.solver = fe.NonlinearVariationalSolver(
            self.problem, solver_parameters = solver_parameters)
        