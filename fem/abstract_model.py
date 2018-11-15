""" **abstract_model.py**
provides an abstract class on which to base finite element models.
"""
import firedrake as fe
import abc


class AbstractModel(metaclass = abc.ABCMeta):
    """ An abstract class on which to base finite element models. """
    def __init__(self):
        
        self.quadrature_degree = None
        
        self.set_mesh()
        
        self.set_element()
        
        self.function_space = fe.FunctionSpace(self.mesh, self.element)
        
        self.init_solution()
    
    @abc.abstractmethod
    def set_mesh(self):
        """ Redefine this to return a `fe.Mesh`.
        """
    
    @abc.abstractmethod
    def set_element(self):
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
    
        r = self.weak_form_residual()*fe.dx(degree = self.quadrature_degree)
        
        u = self.solution
        
        bcs = self.dirichlet_boundary_conditions()
        
        problem = fe.NonlinearVariationalProblem(r, u, bcs, fe.derivative(r, u))
        
        solver = fe.NonlinearVariationalSolver(
            problem, solver_parameters = solver_parameters)
        
        self.solver = solver
        
    def unit_vectors(self):
        
        dim = self.mesh.ufl_dim()
        
        return tuple([fe.unit_vector(i, dim) for i in range(dim)])
        