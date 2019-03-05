""" Contains the Model class """
import firedrake as fe
import pathlib
import matplotlib.pyplot as plt
import fempy.time_discretization
import fempy.output


default_solver_parameters = {
    "snes_type": "newtonls",
    "snes_monitor": True,
    "ksp_type": "preonly", 
    "pc_type": "lu", 
    "mat_type": "aij",
    "pc_factor_mat_solver_type": "mumps"}

    
""" Time dependence """
def time_discrete_terms(solutions, timestep_size):
    
    time_discrete_terms = [
        fempy.time_discretization.bdf(
            [fe.split(solutions[n])[i] for n in range(len(solutions))],
            order = len(solutions) - 1,
            timestep_size = timestep_size)
        for i in range(len(fe.split(solutions[0])))]
        
    if len(time_discrete_terms) == 1:
    
        time_discrete_terms = time_discrete_terms[0]
        
    else:
    
        time_discrete_terms = time_discrete_terms

    return time_discrete_terms
    
    
""" Helpers """
def unit_vectors(mesh):
    
    dim = mesh.geometric_dimension()
    
    return tuple([fe.unit_vector(i, dim) for i in range(dim)])

    
""" Output """    
def plotvars(solution):
    
    subscripts, functions = enumerate(solution.split())
    
    labels = [r"$w_{0}$".format(i) for i in subscripts]
    
    filenames = ["w{0}".format(i) for i in subscripts]
    
    return functions, labels, filenames
    
    
""" The main class """    
class Model(object):
    """ A class on which to base finite element models """
    def __init__(self, 
            mesh, 
            element, 
            variational_form_residual,
            dirichlet_boundary_conditions,
            initial_values,
            solver_parameters = default_solver_parameters,
            time_stencil_size = 2,
            quadrature_degree = None):
        
        self.mesh = mesh
        
        self.element = element
        
        self.function_space = fe.FunctionSpace(mesh, element)
        
        self.integration_measure = fe.dx(degree = quadrature_degree)
        
        """ Time dependence """
        self.solutions = [fe.Function(self.function_space) 
            for i in range(time_stencil_size)]
            
        self.solution = self.solutions[0]
        
        self.time = fe.Constant(0.)
        
        self.timestep_size = fe.Constant(1.)
        
        self.time_tolerance = 1.e-8
        
        self.initial_values = initial_values(model = self)
        
        self.assign_initial_values_to_solutions()
        
        """ Construct the variational problem and solver """
        self.variational_form = variational_form_residual(
                model = self,
                solution = self.solution)*self.integration_measure
                
        self.dirichlet_boundary_conditions = \
            dirichlet_boundary_conditions(model = self)
                
        self.solver_parameters = solver_parameters
        
        self.problem, self.solver = self.reset_problem_and_solver()
        
        """ Output """
        self.output_directory_path = pathlib.Path("output/")
        
        self.snes_iteration_counter = 0
    
    def reset_problem_and_solver(self):
    
        self.problem = fe.NonlinearVariationalProblem(
            F = self.variational_form,
            u = self.solution,
            bcs = self.dirichlet_boundary_conditions,
            J = fe.derivative(self.variational_form, self.solution))
        
        self.solver = fe.NonlinearVariationalSolver(
            problem = self.problem,
            solver_parameters = self.solver_parameters)
            
        return self.problem, self.solver
    
    def solve(self):
    
        self.solver.solve()
        
        self.snes_iteration_counter += self.solver.snes.getIterationNumber()
        
        return self.solution, self.snes_iteration_counter
    
    """ Time dependence """
    def push_back_solutions(self):
        
        for i in range(len(self.solutions[1:])):
        
            self.solutions[-(i + 1)].assign(
                self.solutions[-(i + 2)])
                
        return self.solutions
        
    def run(self,
            endtime,
            report = True,
            write_solution = False,
            plot = False,
            assign_initial_values_to_solutions = True,
            quiet = False):
            
        assert(len(self.solutions) > 1)
        
        self.output_directory_path.mkdir(
            parents = True, exist_ok = True)
        
        solution_filepath = self.\
            output_directory_path.joinpath("solution").with_suffix(".pvd")
        
        if write_solution:
        
            solution_file = fe.File(str(solution_filepath))
        
        if assign_initial_values_to_solutions:
        
            for solution in self.solutions:
        
                solution.assign(self.initial_values)
            
        if report:
            
            fempy.output.report(self, write_header = True)
        
        if write_solution:
        
            write_solution(solution_file)
        
        if plot:
            
            fempy.output.plot(self)
            
        while self.time.__float__() < (
                endtime - self.time_tolerance):
            
            self.time.assign(self.time + self.timestep_size)
                
            self.solution, _ = self.solve()
            
            if report:
            
                fempy.output.report(self, write_header = True)
                
            if write_solution:
        
                fempy.output.write_solution(self, solution_file)
                
            if plot:
            
                fempy.output.plot(self)
            
            self.solutions = self.push_back_solutions()
            
            if not quiet:
            
                print("Solved at time t = {0}".format(self.time.__float__()))
                
        return self.solutions, self.time

    """ Helpers """
    def assign_parameters(self, parameters):
    
        for key, value in parameters.items():
        
            attribute = getattr(self, key)
            
            if type(attribute) is type(fe.Constant(0.)):
            
                attribute.assign(value)
                
            else:
            
                setattr(self, key, value)
                
        return self
                
    def assign_initial_values_to_solutions(self):
    
        for solution in self.solutions:
        
            solution.assign(self.initial_values)
            
        return self.solutions
        