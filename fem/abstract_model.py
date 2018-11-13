""" **abstract_model.py**
provides an abstract class on which to base finite element models.
"""
import firedrake as fe
import abc


class AbstractModel(metaclass = abc.ABCMeta):
    """ An abstract class on which to base finite element models. """
    def __init__(self, 
            mesh, 
            dirichlet_boundary_conditions = [
                {"subspace": None, "value": 0., "subdomain": "on_boundary"},],
            quadrature_degree = 2,
            solver_parameters = {
                "ksp_type": "preonly", 
                "pc_type": "lu", 
                "mat_type": "aij",
                "pc_factor_mat_solver_type": "mumps"}):
        
        self.mesh = mesh
        
        element = self.element()
        
        V = fe.FunctionSpace(mesh, element)
        
        u = fe.Function(V)
        
        self.solution = u
        
        
        bcs = []
        
        for i, bc in enumerate(dirichlet_boundary_conditions):
        
            if bc["subspace"] is None:
            
                V_i = V
                
            else:
            
                V_i = V.sub(i)
                
            bcs.append(fe.DirichletBC(V_i, bc["value"], bc["subdomain"]))
        
        
        r = self.weak_form_residual()*fe.dx(degree = quadrature_degree)
        
        problem = fe.NonlinearVariationalProblem(
            r, u, bcs, fe.derivative(r, u))
        
        solver = fe.NonlinearVariationalSolver(
            problem, solver_parameters = solver_parameters)
        
        self.solver = solver
        
    @abc.abstractmethod
    def element(self):
        """ Redefine this 
        to return a `fe.FiniteElement` or `fe.MixedElement`.
        """
        
    @abc.abstractmethod
    def weak_form_residual(self):
        """ Redefine this 
        to return a `fe.NonlinearVariationalForm`.
        """
        