"""Provides a class for constructing simulations based on Firedrake.

Simulations proceed forward in time by solving 
a sequence of Initial Boundary Values Problems (IBVP's).

Using the Firedrake framework, 
the PDE's are discretized in space with Finite Elements (FE).

The symbolic capabilities of Firedrake are used to 
automatically implement backward difference formula (BDF) time 
discretizations and to automatically linearize nonlinear problems 
with Newton's method.

Nonlinear and linear solvers are provided by PETSc
and are accessed via the Firedrake interface.

This module imports `firedrake` as `fe` and its documentation writes
`fe` instead of `firedrake`.
"""
import typing
import pathlib
import ufl
import firedrake as fe
import sapphire.time_discretization
import sapphire.output


class Simulation(sapphire.output.ObjectWithOrderedDict):
    """A PDE-based simulation using the Firedrake framework.

    The PDE's are discretized in space using finite elements 
    and in time using backward difference formulas.
    
    Implementing a simulation requires at least instantiating this 
    class and calling the instance's `run` method.
    
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
                This also determines the number of stored solutions.
                Must be greater than zero.
                Defaults to 2. Set to 1 for steady state problems.
                Increase for higher-order time accuracy.
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
        
        self.quadrature_degree = quadrature_degree
        
        self.solver_parameters = solver_parameters
        
        self.timestep_size = fe.Constant(timestep_size)
        
        self.time = fe.Constant(initial_time)
        
        
        # Solution components
        self.function_space = fe.FunctionSpace(mesh, element)
        
        assert(time_stencil_size > 0)
        
        self.solutions = [
            fe.Function(self.function_space, name = solution_name) 
            for i in range(time_stencil_size)]
            
        self.solution = self.solutions[0]
        
        self.backup_solution = fe.Function(self.solution)
        
        self.initial_values = initial_values(sim = self)
        
        for solution in self.solutions:
            # Assume that the initial solution is at a steady state.
            solution.assign(self.initial_values)
            
            
        # Problem components
        self.weak_form_residual = weak_form_residual(
                sim = self,
                solution = self.solution)
                
        self.dirichlet_boundary_conditions = \
            dirichlet_boundary_conditions(sim = self)
            
            
        # Output controls
        self.output_directory_path = pathlib.Path(output_directory_path)
        
        self.output_directory_path.mkdir(parents = True, exist_ok = True)
        
        self.solution_file = None
        
        self.plotvars = None
        
        self.postprocessing_function_space = \
            fe.FunctionSpace(
                self.mesh,
                fe.FiniteElement("P", self.mesh.ufl_cell(), 1))
        
        self.snes_iteration_count = 0
        
    def run(self,
            endtime: float,
            plot: bool = False,
            write_initial_outputs: bool = True,
            endtime_tolerance: float = 1.e-8,
            solve: typing.Callable = None) \
            -> (typing.List[fe.Function], float):
        """Run simulation forward in time.
        
        Args:
            endtime (float): Run until reaching this time.
            plot (bool): Write plots if True. Defaults to False.
                Writing the plots to disk can in some cases dominate
                the processing time. Additionally, much more data
                is generated, requiring more disk storage space.
            write_initial_outputs (bool): Write for initial values
                before solving the first time step. Default to True.
                You may want to set this to False if, for example, you
                are calling `run` repeatedly with later endtimes.
                In such a case, the initial values are the same as 
                the previously computed solution, and so they should
                not be written again.
            endtime_tolerance (float): Allows endtime to be only
                approximately reached. This is larger than a 
                typical floating point comparison tolerance
                because errors accumulate between timesteps.
            solve (callable): This is called to solve each time step.
                By default, this will be set to `self.solve`.
        """
        if write_initial_outputs:
        
            self.write_outputs(write_headers = True, plot = plot)
        
        if solve is None:
        
            solve = self.solve
        
        while self.time.__float__() < (endtime - endtime_tolerance):
            
            self.time = self.time.assign(self.time + self.timestep_size)
            
            self.solution = solve()
            
            print("Solved at time t = {}".format(self.time.__float__()))
            
            self.write_outputs(write_headers = False, plot = plot)
            
            self.solutions = self.push_back_solutions()
            
        return self.solutions, self.time
        
    def solve(self) -> fe.Function:
        """Set up the problem and solver, and solve.
        
        This is a JIT (just in time), ensuring that the problem and 
        solver setup are up-to-date before calling the solver.
        All compiled objects are cached, so the JIT problem and solver 
        setup does not have any significant performance overhead.
        """
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
    
    def push_back_solutions(self) -> typing.List[fe.Function]:
        """Push back listed solutions from discrete times.
        
        Only enough solutions are stored for the time discretization.
        Advancing the simulation forward in time requires re-indexing
        the solutions.
        """
        for i in range(len(self.solutions[1:])):
        
            self.solutions[-(i + 1)].assign(
                self.solutions[-(i + 2)])
                
        return self.solutions
        
    def postprocess(self) -> 'Simulation':
        """ This is called by `write_outputs` before writing. 
        
        Redefine this to add post-processing.
        """
        return self
        
    def kwargs_for_writeplots(self) -> dict:
        """Return kwargs needed for `sappphire.outupt.writeplots`.
        
        By default, no plots are made.
        This must be redefined to return a dict 
        if `run` is called with `plot = True`.
        """
        return None
        
    def write_outputs(self, 
            write_headers: bool, 
            plot: bool = False):
        """Write all outputs.
        
        This creates or appends the CSV report, 
        writes the latest solution, and plots (in 1D/2D case).
        Redefine this to control outputs.
        
        Args:
            write_headers (bool): Write header line to report if True.
                You may want to set this to False, for example, if the 
                header has already been written.
            plot (bool): Write plots if True.
        """
        if self.solution_file is None:
            
            solution_filepath = self.output_directory_path.joinpath(
                "solution").with_suffix(".pvd")
            
            self.solution_file = fe.File(str(solution_filepath))
        
        self = self.postprocess()
        
        sapphire.output.report(sim = self, write_header = write_headers)
        
        sapphire.output.write_solution(sim = self, file = self.solution_file)
        
        if plot:
            
            if self.mesh.geometric_dimension() < 3:
                
                sapphire.output.writeplots(
                    **self.kwargs_for_writeplots(),
                    time = self.time.__float__(),
                    outdir_path = self.output_directory_path)
                
            elif self.mesh.geometric_dimension() == 3:
                # This could be done with VTK and PyVista, but VTK can be a
                # difficult dependency. It may be best to run a separate 
                # program for generating 3D plots from the solution files.
                raise NotImplementedError()
                
    def unit_vectors(self) -> typing.Tuple[ufl.tensors.ListTensor]:
        """Returns the spatial unit vectors in each dimension."""
        return unit_vectors(self.mesh)
        
        
def unit_vectors(mesh) -> typing.Tuple[ufl.tensors.ListTensor]:
    """Returns the mesh's spatial unit vectors in each dimension.
    
    Args:
        mesh (fe.Mesh): The mesh for the spatial discretization.
    """
    dim = mesh.geometric_dimension()
    
    return tuple([fe.unit_vector(i, dim) for i in range(dim)])
    
    
def time_discrete_terms(
        solutions: typing.List[fe.Function],
        timestep_size: fe.Constant) \
        -> typing.Union[
            ufl.core.operator.Operator,
            typing.List[ufl.core.operator.Operator]]:
    """Returns backward difference time discretization.
    
    The backward difference formula's stencil size is determine by the
    number of solutions provided, i.e. `len(solutions)`.
    For example, if `len(solutions == 3)`, then the second-order BDF2
    method will be used, because it involves solutions at three 
    discrete times.
    
    The return type depends on whether or not the solution is based on
    a mixed finite element. For mixed finite elements, a list of time
    discrete terms will be returned, each item corresponding to one of 
    the sub-elements of the mixed element. Otherwise, a single term
    will be returned.
    """
    """
    The return type design choice was made, rather than always
    returning a list (e.g. with only one item if not using a mixed 
    element), so that it would be more intuitive when not using mixed 
    elements.
    """
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
    