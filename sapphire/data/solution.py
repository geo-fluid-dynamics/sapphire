"""Solution data module"""
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Union, Tuple, Dict, List, Any
from firedrake.functionspace import FunctionSpace
from firedrake.functionspaceimpl import WithGeometry
from sapphire.data.mesh import Mesh
from firedrake import Constant, FiniteElement, VectorElement, MixedElement,  Function, TestFunction, TestFunctions, split, unit_vector


@dataclass  # pylint: disable=too-many-instance-attributes
class Solution:
    """Solution data calss

    This includes a solution and its associated time, post processed functions, and other auxiliary information."""
    mesh: Mesh

    element: Union[FiniteElement, VectorElement, MixedElement]

    component_names: Tuple[str]

    ufl_constants: Union[Dict[str, Constant], Tuple[Constant], None] = None

    quadrature_degree: Union[int, None] = None

    time: Union[float, None] = None

    geometric_dimension: int = field(init=False)

    unit_vectors: Any = field(init=False)

    function: Function = field(init=False)

    component_count: int = field(init=False)

    function_space: Union[Any, WithGeometry] = field(init=False)

    ufl_fields: Tuple[Constant] = field(init=False)

    subfunctions: Tuple[Function] = field(init=False)

    test_functions: Tuple[TestFunction] = field(init=False)

    snes_cumulative_iteration_count: int = field(init=False)

    continuation_history: List[Tuple[str, float, int]] = field(init=False)
    """ List of triplets with continuation parameter name, continuation parameter value, and SNES iteration count """

    checkpoint_index: int = field(init=False)

    post_processed_functions: List[Function] = field(init=False)

    def __post_init__(self):

        if isinstance(self.ufl_constants, dict):

            _ufl_constants = self.ufl_constants.copy()

            for key in self.ufl_constants:

                _ufl_constants[key] = Constant(self.ufl_constants[key])

            self.ufl_constants = namedtuple('UFLConstants', self.ufl_constants.keys())(**_ufl_constants)

        self.geometric_dimension = self.mesh.geometry.geometric_dimension()

        self.unit_vectors = tuple(unit_vector(i, self.geometric_dimension) for i in range(self.geometric_dimension))

        self.function = Function(FunctionSpace(self.mesh.geometry, self.element))

        self.component_count = len(self.function.split())

        if len(self.component_names) != self.component_count:

            raise Exception("A field name must be provided for every subspace.")

        self.function_space = self.function.function_space()

        self.ufl_fields = namedtuple('UFLFields', self.component_names)(*split(self.function))

        self.subfunctions = namedtuple('SubFunctions', self.component_names)(*self.function.split())

        self.function_subspaces = namedtuple('FunctionSubspaces', self.component_names)(*[self.function.function_space().sub(i) for i in range(self.component_count)])

        self.test_functions = namedtuple('TestFunctions', self.component_names)(*TestFunctions(self.function.function_space()))

        self.continuation_history = []

        self.post_processed_functions = []

        self.checkpoint_index = 0

        self.snes_cumulative_iteration_count = 0
