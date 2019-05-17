""" Contains the Simulation class """
import pathlib
import firedrake as fe
import sapphire.time_discretization
import sapphire.output


time_tolerance = 1.e-8

class Simulation(sapphire.output.ObjectWithOrderedDict):
    """ A class on which to base finite element simulations """
    def __init__(self, 
            mesh, 
            element, 
            variational_form_residual,
            dirichlet_boundary_conditions,
            initial_values,
            quadrature_degree = None,
            time_dependent = True,
            time_stencil_size = 2,
            output_directory_path = "output/"):
        
        self.mesh = mesh
        
        self.element = element
        
        self.function_space = fe.FunctionSpace(mesh, element)
        
        self.quadrature_degree = quadrature_degree
        
        
        self.solutions = [fe.Function(self.function_space) 
            for i in range(time_stencil_size)]
            
        self.solution = self.solutions[0]
        
        self.backup_solution = fe.Function(self.solution)
        
        
        if time_dependent:
            
            assert(time_stencil_size > 1)
            
            self.time = fe.Constant(0.)
            
            self.timestep_size = fe.Constant(1.)
            
        else:
        
            self.time = None
        
            self.timestep_size = None
            
            
        self.output_directory_path = pathlib.Path(output_directory_path)
        
        self.solution_file = None
        
        self.plotvars = None
        
        
        self.initial_values = initial_values(sim = self)
        
        for solution in self.solutions:
        
            solution.assign(self.initial_values)
        
        
        self.variational_form_residual = variational_form_residual(
                sim = self,
                solution = self.solution)
                
        self.dirichlet_boundary_conditions = \
            dirichlet_boundary_conditions(sim = self)
        
        
        self.snes_iteration_count = 0
        
    def solve(self,
            parameters = {
                "snes_type": "newtonls",
                "snes_monitor": None,
                "ksp_type": "preonly", 
                "pc_type": "lu", 
                "mat_type": "aij",
                "pc_factor_mat_solver_type": "mumps"}):

        problem = fe.NonlinearVariationalProblem(
            F = self.variational_form_residual,
            u = self.solution,
            bcs = self.dirichlet_boundary_conditions,
            J = fe.derivative(self.variational_form_residual, self.solution))
            
        solver = fe.NonlinearVariationalSolver(
            problem = problem,
            solver_parameters = parameters)
            
        solver.solve()
        
        self.snes_iteration_count += solver.snes.getIterationNumber()
        
        return self.solution
    
    def push_back_solutions(self):
        
        for i in range(len(self.solutions[1:])):
        
            self.solutions[-(i + 1)].assign(
                self.solutions[-(i + 2)])
                
        return self.solutions
        
    def postprocess(self):
    
        return self
        
    def write_outputs(self, write_headers, plotvars = None):
        
        self.output_directory_path.mkdir(
            parents = True, exist_ok = True)
        
        if self.solution_file is None:
            
            solution_filepath = self.output_directory_path.joinpath(
                "solution").with_suffix(".pvd")
            
            self.solution_file = fe.File(str(solution_filepath))
        
        self = self.postprocess()
        
        sapphire.output.report(sim = self, write_header = write_headers)
        
        sapphire.output.write_solution(sim = self, file = self.solution_file)
        
        sapphire.output.plot(sim = self, plotvars = plotvars)
        
    def run(self,
            endtime,
            solve = None,
            write_initial_outputs = True):
        
        if write_initial_outputs:
        
            self.write_outputs(write_headers = True)
        
        if solve is None:
        
            solve = self.solve
        
        while self.time.__float__() < (endtime - time_tolerance):
            
            self.time.assign(self.time + self.timestep_size)
            
            self.solution = solve()
            
            print("Solved at time t = {0}".format(self.time.__float__()))
            
            self.write_outputs(write_headers = False)
            
            self.solutions = self.push_back_solutions()
            
        return self.solutions, self.time
        
    def assign_parameters(self, parameters):
    
        for key, value in parameters.items():
        
            attribute = getattr(self, key)
            
            if type(attribute) is type(fe.Constant(0.)):
            
                attribute.assign(value)
                
            else:
            
                setattr(self, key, value)
                
        return self
        
    def unit_vectors(self):
    
        return unit_vectors(self.mesh)
        
        
def unit_vectors(mesh):
    
    dim = mesh.geometric_dimension()
    
    return tuple([fe.unit_vector(i, dim) for i in range(dim)])
    
    
def time_discrete_terms(solutions, timestep_size):
    
    time_discrete_terms = [
        sapphire.time_discretization.bdf(
            [fe.split(solutions[n])[i] for n in range(len(solutions))],
            timestep_size = timestep_size)
        for i in range(len(fe.split(solutions[0])))]
        
    if len(time_discrete_terms) == 1:
    
        time_discrete_terms = time_discrete_terms[0]
        
    else:
    
        time_discrete_terms = time_discrete_terms

    return time_discrete_terms
    