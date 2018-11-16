""" An abstract class on which to base finite element models """
import firedrake as fe
import abc


class Model(metaclass = abc.ABCMeta):
    """ An abstract class on which to base finite element models. """
    def __init__(self):
        
        self.init_mesh()
        
        self.init_element()
        
        self.function_space = fe.FunctionSpace(self.mesh, self.element)
        
        self.init_solution()
        
        self.integration_measure = fe.dx
    
    @abc.abstractmethod
    def init_mesh(self):
        """ Redefine this to return a `fe.Mesh`.
        """
    
    @abc.abstractmethod
    def init_element(self):
        """ Redefine this to return a `fe.FiniteElement` or `fe.MixedElement`.
        """
        
    @abc.abstractmethod
    def weak_form_residual(self):
        """ Redefine this to return a `fe.NonlinearVariationalForm`.
        """
    
    def init_solution(self):
    
        self.solution = fe.Function(self.function_space)
    
    def dirichlet_boundary_conditions(self):
        """ Optionally redefine this to return a tuple of `fe.DirichletBC`.
        """
        return None
        
    def solve(self,
            solver_parameters = {
                "ksp_type": "preonly", 
                "pc_type": "lu", 
                "mat_type": "aij",
                "pc_factor_mat_solver_type": "mumps"}):
    
        r = self.weak_form_residual()*self.integration_measure
        
        u = self.solution
        
        bcs = self.dirichlet_boundary_conditions()
        
        problem = fe.NonlinearVariationalProblem(r, u, bcs, fe.derivative(r, u))
        
        solver = fe.NonlinearVariationalSolver(
            problem, solver_parameters = solver_parameters)
        
        solver.solve()
    
    def unit_vectors(self):
        
        dim = self.mesh.geometric_dimension()
        
        return tuple([fe.unit_vector(i, dim) for i in range(dim)])
        