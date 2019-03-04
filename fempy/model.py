""" Contains the Model class """
import firedrake as fe
import pathlib
import matplotlib.pyplot as plt


def unit_vectors(mesh):
    
    dim = mesh.geometric_dimension()
    
    return tuple([fe.unit_vector(i, dim) for i in range(dim)])
        

class Model(object):
    """ A class on which to base finite element models """
    def __init__(self, mesh, element, quadrature_degree):
        
        self.mesh = mesh
        
        self.element = element
        
        self.quadrature_degree = quadrature_degree
        
        self.init_function_space()
        
        self.init_solution()
        
        self.init_integration_measure()
        
        self.init_weak_form_residual()
        
        self.init_dirichlet_boundary_conditions()
        
        self.init_problem()
        
        self.init_solver()
        
        self.quiet = False
        
        self.output_directory_path = pathlib.Path("output/")
        
        self.snes_iteration_counter = 0
    
    def init_weak_form_residual(self):
        """ Redefine this to set `self.weak_form_residual` 
        to a `fe.NonlinearVariationalForm`.
        """
        assert(False)
    
    def init_function_space(self):
    
        self.function_space = fe.FunctionSpace(self.mesh, self.element)
    
    def init_solution(self):
    
        self.solution = fe.Function(self.function_space)
        
    def init_dirichlet_boundary_conditions(self):
        """ Optionallay redefine this 
        to set `self.dirichlet_boundary_conditions`
        to a tuple of `fe.DirichletBC` """
        self.dirichlet_boundary_conditions = None
        
    def init_integration_measure(self):

        self.integration_measure = fe.dx(degree = self.quadrature_degree)
        
    def init_problem(self):
    
        r = self.weak_form_residual*self.integration_measure
        
        u = self.solution
        
        self.problem = fe.NonlinearVariationalProblem(
            r, u, self.dirichlet_boundary_conditions, fe.derivative(r, u))
        
    def init_solver(self, solver_parameters = {
            "snes_type": "newtonls",
            "snes_monitor": True,
            "ksp_type": "preonly", 
            "pc_type": "lu", 
            "mat_type": "aij",
            "pc_factor_mat_solver_type": "mumps"}):
        
        self.solver = fe.NonlinearVariationalSolver(
            self.problem, solver_parameters = solver_parameters)
    
    def assign_parameters(self, parameters):
    
        for key, value in parameters.items():
        
            attribute = getattr(self, key)
            
            if type(attribute) is type(fe.Constant(0.)):
            
                attribute.assign(value)
                
            else:
            
                setattr(self, key, value)
    
    def solve(self):
    
        self.solver.solve()
        
        self.snes_iteration_counter += self.solver.snes.getIterationNumber()
        
    def unit_vectors(self):
    
        return unit_vectors(self.mesh)
        
    def plotvars(self):
    
        subscripts, functions = enumerate(model.solution.split())
        
        labels = [r"$w_{0}$".format(i) for i in subscripts]
        
        filenames = ["w{0}".format(i) for i in subscripts]
        
        return functions, labels, filenames
        