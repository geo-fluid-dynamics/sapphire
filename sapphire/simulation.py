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


class Simulation:
    """A PDE-based simulation using the Firedrake framework.

    The PDE's are discretized in space using finite elements 
    and in time using backward difference formulas.
    
    Implementing a simulation requires at least instantiating this 
    class and calling the instance's `run` method.
    """
    
    def __init__(self,
            solution: fe.Function,
            time: float = 0.,
            time_stencil_size: int = 2,
            timestep_size: float = 1.,
            quadrature_degree: int = None,
            solver_parameters: dict = {
                "snes_type": "newtonls",
                "snes_monitor": None,
                "ksp_type": "preonly", 
                "pc_type": "lu", 
                "mat_type": "aij",
                "pc_factor_mat_solver_type": "mumps"},
            output_directory_path: str = "output/",
            fieldnames: typing.Iterable[str] = None):
        """
        Instantiating this class requires enough information to fully 
        specify the FE spatial discretization and weak form residual.
        boundary conditions, and initial values. All of these required
        arguments are Firedrake objects used according to Firedrake
        conventions.
        
        Backward Difference Formula time discretizations are
        automatically implemented. To use a different time
        discretization, inherit this class and redefine 
        `time_discrete_terms`.
        
        Args:
            solution: Solution for a single time step.
                As a `fe.Function`, this also defines the 
                mesh, element, and solution function space.
            time: The initial time.
            time_stencil_size: The number of solutions at 
                discrete times used for approximating time derivatives.
                This also determines the number of stored solutions.
                Must be greater than zero.
                Defaults to 2. Set to 1 for steady state problems.
                Increase for higher-order time accuracy.
            timestep_size: The size of discrete time steps.
                Defaults to 1.
                Higher order time discretizations are assumed to use
                a constant time step size.
            quadrature_degree: The quadrature degree used for
                numerical integration.
                Defaults to `None`, in which case Firedrake will 
                automatically choose a suitable quadrature degree.
            solver_parameters: The solver parameters dictionary
                which Firedrake uses to configure PETSc.
            output_directory_path: String that will be converted
                to a Path where output files will be written.
                Defaults to "output/".
            fieldnames: A list of names for the components of `solution`.
                Defaults to `None`.
                These names can be used when indexing solutions that are split
                either by `firedrake.split` or `firedrake.Function.split`.
                If not `None`, then the `dict` `self.solution_fields` will be created.
                The `dict` will have two items for each field,
                containing the results of either splitting method.
                The results of `firedrake.split` will be suffixed with "_ufl".
        """
        assert(time_stencil_size > 0)
        
        
        self.fieldcount = len(solution.split())
        
        if fieldnames is None:
        
            fieldnames = ["w_{}" for i in range(self.fieldcount)]
        
        assert(len(fieldnames) == self.fieldcount)
        
        self.fieldnames = fieldnames
        
        
        self.solution = solution
        
        self.time = fe.Constant(time)
        
        
        self.solution_space = self.solution.function_space()
        
        self.mesh = self.solution_space.mesh()
        
        self.unit_vectors = unit_vectors(self.mesh) 
        
        self.element = self.solution_space.ufl_element()
        
        self.timestep_size = fe.Constant(timestep_size)
        
        self.quadrature_degree = quadrature_degree
        
        self.dx = fe.dx(degree = self.quadrature_degree)
        
        self.solver_parameters = solver_parameters
        
        
        initial_values = self.initial_values()
        
        if initial_values is not None:
        
            self.solution = self.solution.assign(initial_values)
        
        
        # States for time dependent simulation and checkpointing
        self.solutions = [self.solution,]
        
        self.times = [self.time,]
        
        self.state = {
            "solution": self.solution,
            "time": self.time,
            "index": 0}
            
        self.states = [self.state,]
        
        for i in range(1, time_stencil_size):
        
            self.solutions.append(fe.Function(self.solution))
            
            self.times.append(fe.Constant(self.time - i*timestep_size))
        
            self.states.append({
                "solution": self.solutions[i],
                "time": self.times[i],
                "index": -i})
        
        
        # Continuation helpers
        self.backup_solution = fe.Function(self.solution)
        
        
        # Mixed solution indexing helpers
        self.solution_fields = {}
        
        self.solution_subfunctions = {}
        
        self.test_functions = {}
        
        self.time_discrete_terms = {}
        
        self.solution_subspaces = {}
        
        for name, field, field_pp, testfun, timeterm in zip(
                fieldnames,
                fe.split(self.solution),
                self.solution.split(),
                fe.TestFunctions(self.solution_space),
                time_discrete_terms(
                    solutions = self.solutions,
                    timestep_size = self.timestep_size)):
            
            self.solution_fields[name] = field
            
            self.solution_subfunctions[name] = field_pp
            
            self.test_functions[name] = testfun
            
            self.time_discrete_terms[name] = timeterm
            
            self.solution_subspaces[name] = self.solution_space.sub(
                fieldnames.index(name))
                
                
        # Output controls
        self.output_directory_path = pathlib.Path(output_directory_path)
        
        self.output_directory_path.mkdir(parents = True, exist_ok = True)
        
        self.vtk_solution_file = None
        
        self.plotvars = None
        
        self.snes_iteration_count = 0
        
    def run(self,
            endtime: float,
            write_checkpoints: bool = True,
            write_vtk_solutions: bool = False,
            write_plots: bool = False,
            write_initial_outputs: bool = True,
            endtime_tolerance: float = 1.e-8,
            solve: typing.Callable = None) \
            -> (typing.List[fe.Function], float):
        """Run simulation forward in time.
        
        Args:
            endtime (float): Run until reaching this time.
            write_vtk_solutions (bool): Write checkpoints if True.
            write_vtk_solutions (bool): Write solutions to VTK if True.
            write_plots (bool): Write plots if True.
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
        
            self.write_outputs(
                headers = True,
                checkpoint = write_checkpoints,
                vtk = write_vtk_solutions,
                plots = write_plots)
        
        if solve is None:
        
            solve = self.solve
        
        while self.time.__float__() < (endtime - endtime_tolerance):
            
            self.states = self.push_back_states()
            
            self.time = self.time.assign(self.time + self.timestep_size)
            
            self.state["index"] += 1
            
            self.solution = solve()
            
            print("Solved at time t = {}".format(self.time.__float__()))
            
            self.write_outputs(
                headers = False,
                checkpoint = write_checkpoints,
                vtk = write_vtk_solutions,
                plots = write_plots)
            
        return self.states
        
    def solve(self) -> fe.Function:
        """Set up the problem and solver, and solve.
        
        This is a JIT (just in time), ensuring that the problem and 
        solver setup are up-to-date before calling the solver.
        All compiled objects are cached, so the JIT problem and solver 
        setup does not have any significant performance overhead.
        """
        problem = fe.NonlinearVariationalProblem(
            F = self.weak_form_residual(),
            u = self.solution,
            bcs = self.dirichlet_boundary_conditions(),
            J = fe.derivative(self.weak_form_residual(), self.solution))
            
        solver = fe.NonlinearVariationalSolver(
            problem = problem,
            nullspace = self.nullspace(),
            solver_parameters = self.solver_parameters)
            
        solver.solve()
        
        self.snes_iteration_count += solver.snes.getIterationNumber()
        
        return self.solution
    
    def weak_form_residual(self):
        
        raise("This method must be redefined by the derived class.")
    
    def initial_values(self):
    
        return None
    
    def dirichlet_boundary_conditions(self):
        
        return None
    
    def nullspace(self):
        
        return None
    
    def push_back_states(self) -> typing.List[typing.Dict]:
        """Push back states, including solutions, times, and indices.
        
        Sufficient solutions are stored for the time discretization.
        Advancing the simulation forward in time requires re-indexing
        the solutions and times.
        """        
        for i in range(len(self.states[1:])):
            
            rightstate = self.states[-1 - i]
            
            leftstate = self.states[-2 - i]
            
            rightstate["index"] = leftstate["index"]
            
            for key in "solution", "time":
                # Set values of `fe.Function` and `fe.Constant` 
                # with their `assign` methods.
                rightstate[key] = rightstate[key].assign(leftstate[key])
                
        return self.states
        
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
        
    def write_checkpoint(self):
        
        sapphire.output.write_checkpoint(
            states=self.states,
            dirpath=self.output_directory_path,
            filename="checkpoints")
    
    def write_outputs(self, 
            headers: bool,
            checkpoint: bool = True,
            vtk: bool = False,
            plots: bool = False):
        """Write all outputs.
        
        This creates or appends the CSV report, 
        writes the latest solution, and plots (in 1D/2D case).
        Redefine this to control outputs.
        
        Args:
            write_headers (bool): Write header line to report if True.
                You may want to set this to False, for example, if the 
                header has already been written.
            checkpoint (bool): Write checkpoint if True.
            vtk (bool): Write solution to VTK if True.
            plots (bool): Write plots if True.
        """    
        self = self.postprocess()
        
        sapphire.output.report(sim = self, write_header = headers)
        
        if checkpoint:
        
            self.write_checkpoint()
            
        if vtk:
           
            if self.vtk_solution_file is None:
        
                vtk_solution_filepath = self.output_directory_path.joinpath(
                    "solution").with_suffix(".pvd")
                
                self.vtk_solution_file = fe.File(str(vtk_solution_filepath))
            
            sapphire.output.write_solution_to_vtk(
                sim = self, file = self.vtk_solution_file)
                
        if plots:
            
            if self.mesh.geometric_dimension() < 3:
                
                sapphire.output.writeplots(
                    **self.kwargs_for_writeplots(),
                    time = self.time.__float__(),
                    time_index = self.state["index"],
                    outdir_path = self.output_directory_path)
                    
            elif self.mesh.geometric_dimension() == 3:
                # This could be done with VTK and PyVista, but VTK can be a
                # difficult dependency. It may be best to run a separate 
                # program for generating 3D plots from the solution files.
                raise NotImplementedError()

    
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
    """
    This implementation assumes constant time step size.
    Variable time step sizes change the BDF formula 
    for all except first order.
    """
    time_discrete_terms = [
        sapphire.time_discretization.bdf(
            [fe.split(solutions[n])[i] for n in range(len(solutions))],
            timestep_size = timestep_size)
        for i in range(len(solutions[0].split()))]
        
    return time_discrete_terms
    