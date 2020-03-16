""" Provides a PDE-governed Simulation class using Firedrake.

Simulations proceed forward in time by solving 
a sequence of Initial Boundary Values Problems (IBVP's).

Using the Firedrake framework, 
the PDE's are discretized in space with Finite Elements (FE).

Furthermore, the symbolic capabilities of Firedrake are used to 
automatically implement backward difference formula (BDF) time 
discretizations and to automatically linearize nonlinear problems 
with Newton's method.

Nonlinear and linear solvers are provided by PETSc
and are accessed via the Firedrake interface.

This module imports `firedrake` as `fe` and its documentation writes
`fe` instead of `firedrake`.
"""
import pathlib
import logging
import firedrake as fe
import sapphire.time_discretization
import sapphire.output
import typing


time_tolerance = 1.e-8

class Simulation(sapphire.output.ObjectWithOrderedDict):
    """ A PDE simulation that solves an IBVP using FE in space and BDF in time
    
    Implementing a simulation requires at least instantiating this class 
    and calling the instance's `run` method.
    
    This class is derived from `sapphire.output.ObjectWithOrderedDict`
    so that all attributes can be consistently written to a CSV file
    throughout the time-dependent simulation.
    """
    
    def __init__(self, 
            mesh: fe.Mesh, 
            element: typing.Union[fe.FiniteElement, fe.MixedElement],
            weak_form_residual: fe.Form,
            dirichlet_boundary_conditions: typing.List[fe.DirichletBC],
            initial_values: fe.Function,
            quadrature_degree: int = None,
            time_stencil_size: int = 2,
            timestep_size: float = 1.,
            initial_time: float = 0.,
            solver_parameters: dict = {
                "snes_type": "newtonls",
                "snes_monitor": None,
                "ksp_type": "preonly", 
                "pc_type": "lu", 
                "mat_type": "aij",
                "pc_factor_mat_solver_type": "mumps"},
            output_directory_path: str = "output/",
            solution_name: str = None):
        """
        Instantiating this class requires enough information to fully 
        specify the FE spatial discretization, weak form residual,
        boundary conditions, and initial values. All of these required
        arguments are Firedrake objects used according to Firedrake
        conventions.
        
        Backward Difference Formula time discretizations are
        automatically implemented. To use a different time
        discretization, inherit this class and redefine 
        `time_discrete_terms`.
        
        Args:
            mesh (fe.Mesh): The mesh for spatial discretization.
                The spatial degrees of freedom are determined by
                this mesh and the chosen finite element.
            element (fe.FiniteElement or fe.MixedElement):
                The finite element for spatial discretization.
                Firedrake provides a large suite of finite elements.
            weak_form_residual (fe.Form): The weak form residual
                containing the PDE's which govern the simulation.
                The form is defined symbolically using the 
                Unified Form Language (UFL) via Firedrake.
            dirichlet_boundary_conditions (list of fe.DirichletBC):
                The IBVP Dirichlet boundary conditions.
            initial_values (fe.Function): The IBVP initial values
                expressed as a Firedrake `Function`.
                These are the initial values for the first time step.
                For higher order time discretizations, the values are 
                copied backward in time.
                As a simulation proceeds forward in time, using `run`,
                the latest solution(s) will be used as initial values. 
            quadrature_degree (int): The quadrature degree used for
                numerical integration.
                Defaults to `None`, in which case Firedrake will 
                automatically choose a suitable quadrature degree.
            time_stencil_size (int): The number of solutions at 
                discrete times used for approximating time derivatives.
                Defaults to 2. Set to 0 for steady state problems.
            timestep_size (float): The size of discrete time steps.
                Defaults to 1.
                Higher order time discretizations are assumed to use
                a constant time step size.
                Supporting accurate second-order or higher time
                discretizations with variable time step sizes,
                redefine `time_discrete_terms` and compute the
                time step sizes from the solution times.
            initial_time (float): The initial time.
                Defaults to 0.
            solver_parameters (dict): The solver parameters dictionary
                which Firedrake uses to configure PETSc.
            output_directory_path (str): String that will be converted
                to a Path where output files will be written.
                Defaults to "output/".)
            solution_name (str): Overrides the default name that 
                Firedrake otherwise gives the solution Function.
        """
        self.mesh = mesh
        
        self.element = element
        
        self.function_space = fe.FunctionSpace(mesh, element)
        
        self.solutions = [
            fe.Function(self.function_space, name = solution_name) 
            for i in range(time_stencil_size)]
            
        self.solution = self.solutions[0]
        
        self.backup_solution = fe.Function(self.solution)
        
        self.postprocessing_function_space = \
            fe.FunctionSpace(
                self.mesh,
                fe.FiniteElement("P", self.mesh.ufl_cell(), 1))
        
        
        self.output_directory_path = pathlib.Path(output_directory_path)
        
        self.output_directory_path.mkdir(parents = True, exist_ok = True)
        
        self.solution_file = None
        
        self.plotvars = None
        
        
        self.quadrature_degree = quadrature_degree
        
        self.timestep_size = fe.Constant(timestep_size)
        
        self.time = fe.Constant(initial_time)
        
        self.solver_parameters = solver_parameters
        
        self.snes_iteration_count = 0
        
        self.initial_values = initial_values(sim = self)
        
        for solution in self.solutions:
        
            solution.assign(self.initial_values)
        
        
        self.weak_form_residual = weak_form_residual(
                sim = self,
                solution = self.solution)
                
        self.dirichlet_boundary_conditions = \
            dirichlet_boundary_conditions(sim = self)
        
    def solve(self):

        problem = fe.NonlinearVariationalProblem(
            F = self.weak_form_residual,
            u = self.solution,
            bcs = self.dirichlet_boundary_conditions,
            J = fe.derivative(self.weak_form_residual, self.solution))
            
        solver = fe.NonlinearVariationalSolver(
            problem = problem,
            solver_parameters = self.solver_parameters)
            
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
        
        if self.solution_file is None:
            
            solution_filepath = self.output_directory_path.joinpath(
                "solution").with_suffix(".pvd")
            
            self.solution_file = fe.File(str(solution_filepath))
        
        self = self.postprocess()
        
        sapphire.output.report(sim = self, write_header = write_headers)
        
        sapphire.output.write_solution(sim = self, file = self.solution_file)
        
        if self.mesh.geometric_dimension() < 3:
            
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
            
            self.time = self.time.assign(self.time + self.timestep_size)
            
            self.solution = solve()
            
            print("Solved at time t = {0}".format(self.time.__float__()))
            
            self.write_outputs(write_headers = False)
            
            self.solutions = self.push_back_solutions()
            
        return self.solutions, self.time
        
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
    