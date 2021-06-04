"""This module contains Sapphire's core data classes"""
from collections import namedtuple, deque
from dataclasses import dataclass, field
import typing
import sapphire.helpers
import firedrake as fe


@dataclass  # pylint: disable=too-many-instance-attributes
class Solution:
    """This data includes a solution and its associated time, post processed functions, and other auxiliary information."""

    function: fe.Function
    """ Solution for a single time step.
    As a `firedrake.Function`, this also defines the solution function space and therefore the mesh and element.
    """

    function_component_names: typing.Tuple[str]

    ufl_constants: typing.Union[typing.Dict[str, fe.Constant], typing.Tuple[fe.Constant], None] = None

    quadrature_degree: typing.Union[int, None] = None

    time: typing.Union[float, None] = None

    continuation_history: typing.List[typing.Tuple[str, float, int]] = field(init=False)
    """ List of triplets with continuation parameter name, continuation parameter value, and SNES iteration count """

    post_processed_functions: typing.List[fe.Function] = field(init=False)

    checkpoint_index: int = field(init=False)

    snes_cumulative_iteration_count: int = field(init=False)

    function_space: fe.FunctionSpace = field(init=False)

    mesh: fe.Mesh = field(init=False)

    element: typing.Union[fe.FiniteElement, fe.VectorElement, fe.MixedElement] = field(init=False)

    unit_vectors: typing.Any = field(init=False)

    ufl_fields: typing.Tuple[fe.Constant] = field(init=False)

    subfunctions: typing.Tuple[fe.Function] = field(init=False)

    test_functions: typing.Tuple[fe.TestFunction] = field(init=False)

    dim: int = field(init=False)

    def __post_init__(self):

        if isinstance(self.ufl_constants, dict):

            _ufl_constants = self.ufl_constants.copy()

            for key in self.ufl_constants:

                _ufl_constants[key] = fe.Constant(self.ufl_constants[key])

            self.ufl_constants = namedtuple('UFLConstants', self.ufl_constants.keys())(**_ufl_constants)

        self.continuation_history = []

        self.post_processed_functions = []

        self.checkpoint_index = 0

        self.snes_cumulative_iteration_count = 0

        self.function_space = sapphire.helpers.function_space(self.function)

        self.mesh = sapphire.helpers.mesh(self.function_space)

        self.element = sapphire.helpers.element(self.function_space)

        dim = self.mesh.geometric_dimension()

        self.unit_vectors = tuple(fe.unit_vector(i, dim) for i in range(dim))

        component_count = len(self.function.split())

        if len(self.function_component_names) != component_count:

            raise Exception("A field name must be provided for every subspace.")

        self.ufl_fields = namedtuple('UFLFields', self.function_component_names)(*fe.split(self.function))

        self.subfunctions = namedtuple('SubFunctions', self.function_component_names)(*self.function.split())

        self.function_subspaces = namedtuple('FunctionSubspaces', self.function_component_names)(*[self.function_space.sub(i) for i in range(component_count)])

        self.test_functions = namedtuple('TestFunctions', self.function_component_names)(*fe.TestFunctions(self.function_space))

        self.dim = self.mesh.geometric_dimension()


DEFAULT_SOLVER_PARAMTERS = {
    'snes_type': 'newtonls',
    'snes_monitor': None,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'mat_type': 'aij',
    'pc_factor_mat_solver_type': 'mumps'}


@dataclass
class Problem:
    """This data is required to set up and solve a nonlinear problem."""
    solution: Solution

    residual: typing.Callable[[Solution], typing.Any]  # @todo What is the type returned by `* fe.dx`?
    """The residual corresponding to the weak form governing equations.

    This is Callable because the residual must be updated whenever the solution deque rotates to avoid excessive copying of solution function values. """

    dirichlet_boundary_conditions: typing.Tuple[fe.DirichletBC]

    nullspace: typing.Union[fe.MixedVectorSpaceBasis, None] = None

    solver_parameters: dict = None
    """The same that Firedrake uses e.g. for PETSc configuration"""

    def __post_init__(self):

        if self.solver_parameters is None:

            self.solver_parameters = DEFAULT_SOLVER_PARAMTERS


@dataclass
class Simulation:
    """This is the core simulation data class.

    In addition to the problem, this references a solution for each point in the time discretization stencil.
    """
    problem: Problem
    """Only the latest problem is saved."""

    solutions: typing.Deque[Solution]
    """A solution is saved for each point in the time discretization stencil.

    The deque of solutions is arranged with the latest first and the earliest last.
    The latest solution's time is the time that will be solved first.
    The latest solution's initial values will be used as the initial guess for the nonlinear solver.
    """

    def __post_init__(self):

        if len(self.solutions) < 1:

            raise Exception("A simulation must have at least one solution")

        if isinstance(self.solutions, tuple) or isinstance(self.solutions, list):

            self.solutions = deque(self.solutions)

        if not isinstance(self.solutions, deque):

            raise Exception("A simuation must be constructed with either a tuple, list, or deque of solutions.")
