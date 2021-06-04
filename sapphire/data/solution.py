"""Solution data module"""
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Union, Tuple, Dict, List, Any
from sapphire.helpers.upstream import function_space, mesh, element, geometric_dimension
from firedrake import Mesh, FiniteElement, VectorElement, MixedElement, FunctionSpace, Function, TestFunction, TestFunctions, Constant, split, unit_vector


@dataclass  # pylint: disable=too-many-instance-attributes
class Solution:
    """Solution data calss

    This includes a solution and its associated time, post processed functions, and other auxiliary information."""

    function: Function
    """ Solution for a single time step.
    As a `firedrake.Function`, this also defines the solution function space and therefore the mesh and element.
    """

    function_component_names: Tuple[str]

    ufl_constants: Union[Dict[str, Constant], Tuple[Constant], None] = None

    quadrature_degree: Union[int, None] = None

    time: Union[float, None] = None

    continuation_history: List[Tuple[str, float, int]] = field(init=False)
    """ List of triplets with continuation parameter name, continuation parameter value, and SNES iteration count """

    post_processed_functions: List[Function] = field(init=False)

    checkpoint_index: int = field(init=False)

    snes_cumulative_iteration_count: int = field(init=False)

    function_space: FunctionSpace = field(init=False)

    mesh: Mesh = field(init=False)

    element: Union[FiniteElement, VectorElement, MixedElement] = field(init=False)

    unit_vectors: Any = field(init=False)

    ufl_fields: Tuple[Constant] = field(init=False)

    subfunctions: Tuple[Function] = field(init=False)

    test_functions: Tuple[TestFunction] = field(init=False)

    geometric_dimension: int = field(init=False)

    def __post_init__(self):

        if isinstance(self.ufl_constants, dict):

            _ufl_constants = self.ufl_constants.copy()

            for key in self.ufl_constants:

                _ufl_constants[key] = Constant(self.ufl_constants[key])

            self.ufl_constants = namedtuple('UFLConstants', self.ufl_constants.keys())(**_ufl_constants)

        self.continuation_history = []

        self.post_processed_functions = []

        self.checkpoint_index = 0

        self.snes_cumulative_iteration_count = 0

        self.function_space = function_space(self.function)

        self.mesh = mesh(self.function_space)

        self.element = element(self.function_space)

        self.geometric_dimension = geometric_dimension(self.mesh)

        self.unit_vectors = tuple(unit_vector(i, self.geometric_dimension) for i in range(self.geometric_dimension))

        component_count = len(self.function.split())

        if len(self.function_component_names) != component_count:

            raise Exception("A field name must be provided for every subspace.")

        self.ufl_fields = namedtuple('UFLFields', self.function_component_names)(*split(self.function))

        self.subfunctions = namedtuple('SubFunctions', self.function_component_names)(*self.function.split())

        self.function_subspaces = namedtuple('FunctionSubspaces', self.function_component_names)(*[self.function_space.sub(i) for i in range(component_count)])

        self.test_functions = namedtuple('TestFunctions', self.function_component_names)(*TestFunctions(self.function_space))
