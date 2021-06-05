"""Mesh data module"""
from typing import Dict
from dataclasses import dataclass, field
from firedrake import Cell
from firedrake.mesh import MeshGeometry


@dataclass
class Mesh:
    """Mesh data class"""
    geometry: MeshGeometry

    boundaries: Dict[str, int]

    cell: Cell = field(init=False)

    def __post_init__(self):

        self.cell = self.geometry.ufl_cell()
