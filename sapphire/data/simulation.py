"""Simulation data module"""
from collections import deque
from dataclasses import dataclass
from typing import Deque
from sapphire.data.solution import Solution
from sapphire.data.problem import Problem
from sapphire.data.solver import Solver


@dataclass
class Simulation:
    """Simulation data class"""

    solutions: Deque[Solution]
    """Solution data

    A solution is saved for each point in the time discretization stencil.
    The deque of solutions is arranged with the latest first and the earliest last.
    The latest solution's time is the time that will be solved first.
    The latest solution's initial values will be used as the initial guess for the nonlinear solver.
    """

    problem: Problem
    """Data for setting up the nonlinear problem"""

    solver: Solver
    """Data for setting up the nonlinear solver"""

    def __post_init__(self):

        if len(self.solutions) < 1:

            raise Exception("A simulation must have at least one solution")

        if isinstance(self.solutions, tuple) or isinstance(self.solutions, list):

            self.solutions = deque(self.solutions)

        if not isinstance(self.solutions, deque):

            raise Exception("A simuation must be constructed with either a tuple, list, or deque of solutions.")
