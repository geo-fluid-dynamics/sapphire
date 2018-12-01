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
        
        self.init_integration_measure()
        
        self.init_weak_form_residual()
        
        self.init_dirichlet_boundary_conditions()
        
        self.init_problem()
        
        self.init_solver()
    
    @abc.abstractmethod
    def init_mesh(self):
        """ Redefine this to set `self.mesh` to a `fe.Mesh`.
        """
    
    @abc.abstractmethod
    def init_element(self):
        """ Redefine this to set `self.element` 
        to a  `fe.FiniteElement` or `fe.MixedElement`.
        """
        
    @abc.abstractmethod
    def init_weak_form_residual(self):
        """ Redefine this to set `self.weak_form_residual` 
        to a `fe.NonlinearVariationalForm`.
        """
        
    def init_solution(self):
    
        self.solution = fe.Function(self.function_space)
    
    def init_dirichlet_boundary_conditions(self):
        """ Optionallay redefine this 
        to set `self.dirichlet_boundary_conditions`
        to a tuple of `fe.DirichletBC` """
        self.dirichlet_boundary_conditions = None
        
    def init_integration_measure(self):

        self.integration_measure = fe.dx
        
    def init_problem(self):
    
        r = self.weak_form_residual*self.integration_measure
        
        u = self.solution
        
        self.problem = fe.NonlinearVariationalProblem(
            r, u, self.dirichlet_boundary_conditions, fe.derivative(r, u))
        
    def init_solver(self, solver_parameters = {
                "ksp_type": "preonly", 
                "pc_type": "lu", 
                "mat_type": "aij",
                "pc_factor_mat_solver_type": "mumps",
                "snes_monitor": True}):
        
        self.solver = fe.NonlinearVariationalSolver(
            self.problem, solver_parameters = solver_parameters)
    
    def solve(self):
    
        self.solver.solve()
    
    def unit_vectors(self):
        
        dim = self.mesh.geometric_dimension()
        
        return tuple([fe.unit_vector(i, dim) for i in range(dim)])
        