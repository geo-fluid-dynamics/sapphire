import firedrake as fe
import abc


class AbstractModel(metaclass = abc.ABCMeta):

    def __init__(self, mesh, boundary_condition_values):
        
        self.mesh = mesh
        
        element = self.element()
        
        function_space = fe.FunctionSpace(mesh, element)
        
        solution = fe.Function(function_space)
        
        self.solution = solution
        
        try:
        
            iterator = iter(boundary_condition_values)
        
            boundary_conditions = [
                fe.DirichletBC(
                    function_space.sub(i),
                    boundary_condition_values[i],
                    "on_boundary")
                for i, g in enumerate(boundary_condition_values)]
            
        except NotImplementedError as error:
        
            boundary_conditions = fe.DirichletBC(
                function_space, boundary_condition_values, "on_boundary")
        
        F = self.weak_form_residual()
        
        problem = fe.NonlinearVariationalProblem(
            F = F,
            u = solution,
            bcs = boundary_conditions,
            J = fe.derivative(F, solution))
        
        solver = fe.NonlinearVariationalSolver(
            problem, 
            solver_parameters = {
                "ksp_type": "preonly", 
                "pc_type": "lu", 
                "mat_type": "aij",
                "pc_factor_mat_solver_type": "mumps"})
        
        self.solver = solver
        
    @abc.abstractmethod
    def element(self):
        """ Redefine this to return a `fe.MixedElement`. """
        
    @abc.abstractmethod
    def weak_form_residual(self):
        """ Redefine this to return a `fe.NonlinearVariationalForm`. """
        