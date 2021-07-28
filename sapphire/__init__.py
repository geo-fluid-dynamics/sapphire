""" Sapphire (Simulations automatically programmed in Firedrake)

Sapphire is an engine for constructing PDE-based simulations based on the Firedrake framework.

Simulations proceed forward in time by solving a sequence of initial boundary value problems (IBVP's).
Using the Firedrake framework, the PDE's are discretized in space with finite element methods.
The symbolic capabilities of Firedrake are used to automatically implement backward difference formula (BDF) time discretizations and to automatically linearize nonlinear problems with Newton's method.

Nonlinear and linear solvers are provided by PETSc and are accessed via the Firedrake interface.
"""
from sys import version_info
from sapphire.data.mesh import Mesh
from sapphire.data.solution import Solution
from sapphire.data.problem import Problem
from sapphire.data.solver import Solver
from sapphire.data.simulation import Simulation
from sapphire.data.materials import EutecticBinaryAlloy, MATERIALS
from sapphire.time_discretization import bdf
from sapphire.continuation import solve_with_bounded_continuation_sequence, find_working_continuation_parameter_value, ContinuationError, solve_with_timestep_size_continuation
from sapphire.solve import solve
from sapphire.io.report import report
from sapphire.io.checkpoint import write_checkpoint, read_checkpoint
from sapphire.io.plot import plot
from sapphire.run import run


_major, _minor = version_info[:2]

_required_version_message = "Requires Python 3.6 or later"

assert _major >= 3, _required_version_message

if _major == 3:

    assert _minor >= 6, _required_version_message
