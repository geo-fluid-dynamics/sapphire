""" Contains the Model class """
import firedrake as fe
import pathlib
import matplotlib.pyplot as plt
import fempy.time_discretization
import fempy.output


def solve(
        variational_form_residual,
        solution,
        dirichlet_boundary_conditions,
        parameters = {
            "snes_type": "newtonls",
            "snes_monitor": True,
            "ksp_type": "preonly", 
            "pc_type": "lu", 
            "mat_type": "aij",
            "pc_factor_mat_solver_type": "mumps"}):

    problem = fe.NonlinearVariationalProblem(
        F = variational_form_residual,
        u = solution,
        bcs = dirichlet_boundary_conditions,
        J = fe.derivative(variational_form_residual, solution))
        
    solver = fe.NonlinearVariationalSolver(
        problem = problem,
        solver_parameters = parameters)
        
    solver.solve()
    
    return solution, solver.snes.getIterationNumber()
    
    
class Model(object):
    """ A class on which to base finite element models """
    def __init__(self, 
            mesh, 
            element, 
            variational_form_residual,
            dirichlet_boundary_conditions,
            initial_values,
            integration_measure = fe.dx,
            time_stencil_size = 2):
        
        self.mesh = mesh
        
        self.element = element
        
        self.function_space = fe.FunctionSpace(mesh, element)
        
        self.integration_measure = integration_measure
        
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
        self.variational_form_residual = variational_form_residual(
                model = self,
                solution = self.solution)
                
        self.dirichlet_boundary_conditions = \
            dirichlet_boundary_conditions(model = self)
        
        """ Output """
        self.output_directory_path = pathlib.Path("output/")
        
        self.snes_iteration_count = 0
        
    def solve(self, *args, **kwargs):
        
        self.solution, snes_iteration_count = solve(*args,
            variational_form_residual = self.variational_form_residual,
            solution = self.solution,
            dirichlet_boundary_conditions = \
                self.dirichlet_boundary_conditions,
            **kwargs)
           
        self.snes_iteration_count += snes_iteration_count
        
        return self.solution, self.snes_iteration_count
    
    def push_back_solutions(self):
        
        for i in range(len(self.solutions[1:])):
        
            self.solutions[-(i + 1)].assign(
                self.solutions[-(i + 2)])
                
        return self.solutions
        
    def run(self,
            endtime,
            solve = None,
            report = True,
            postprocess = None,
            write_solution = False,
            plot = None):
            
        assert(len(self.solutions) > 1)
        
        if solve is None:
        
            solve = self.solve
        
        self.output_directory_path.mkdir(
            parents = True, exist_ok = True)
        
        solution_filepath = self.\
            output_directory_path.joinpath("solution").with_suffix(".pvd")
        
        if write_solution:
        
            solution_file = fe.File(str(solution_filepath))
            
        if report:
            
            fempy.output.report(
                self, postprocess = postprocess, write_header = True)
        
        if write_solution:
        
            write_solution(solution_file)
        
        if plot:
            
            plot(self)
            
        while self.time.__float__() < (
                endtime - self.time_tolerance):
            
            self.time.assign(self.time + self.timestep_size)
            
            self.solution, self.snes_iteration_count = solve()
            
            if report:
            
                fempy.output.report(
                    self, postprocess = postprocess, write_header = False)
                
            if write_solution:
        
                fempy.output.write_solution(self, solution_file)
                
            if plot:
            
                plot(self, self.solution)
            
            self.solutions = self.push_back_solutions()
            
            print("Solved at time t = {0}".format(self.time.__float__()))
                
        return self.solutions, self.time
        
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
        
        
def unit_vectors(mesh):
    
    dim = mesh.geometric_dimension()
    
    return tuple([fe.unit_vector(i, dim) for i in range(dim)])
    
    
def time_discrete_terms(solutions, timestep_size):
    
    time_discrete_terms = [
        fempy.time_discretization.bdf(
            [fe.split(solutions[n])[i] for n in range(len(solutions))],
            timestep_size = timestep_size)
        for i in range(len(fe.split(solutions[0])))]
        
    if len(time_discrete_terms) == 1:
    
        time_discrete_terms = time_discrete_terms[0]
        
    else:
    
        time_discrete_terms = time_discrete_terms

    return time_discrete_terms
    