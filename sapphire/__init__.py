""" Sapphire (Simulations automatically programmed in Firedrake)

Sapphire is an engine for constructing PDE-based simulations based on
the Firedrake framework.
It provides the `sapphire.Simulation` base class, along with packages 
of general simulations and specific examples.

Simulations proceed forward in time by solving a sequence of 
initial boundary value problems (IBVP's).
Using the Firedrake framework, the PDE's are discretized in space with
finite element methods.
The symbolic capabilities of Firedrake are used to automatically 
implement backward difference formula (BDF) time discretizations and
to automatically linearize nonlinear problems with Newton's method.

Nonlinear and linear solvers are provided by PETSc and are accessed via
the Firedrake interface.
"""
import sys


_major, _minor = sys.version_info[:2]

_required_version_message = "Requires Python 3.6 or later"

assert _major >= 3, _required_version_message

if _major == 3:

    assert _minor >= 6, _required_version_message


from sapphire.simulation import Simulation
