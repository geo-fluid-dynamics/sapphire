"""Verify accuracy of the enthalpy-porosity solver

for convection-coupled phase-change.
"""
import sys
import pathlib
import firedrake as fe 
import sapphire.mms
from sapphire.simulations import enthalpy_porosity as sim_module


pi, sin, exp = fe.pi, fe.sin, fe.exp

def space_verification_solution(sim):
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    ihat, jhat = sim.unit_vectors()
    
    u = exp(0.5)*sin(2.*pi*x)*sin(pi*y)*ihat + \
        exp(0.5)*sin(pi*x)*sin(2.*pi*y)*jhat
    
    p = -sin(pi*x - pi/2.)*sin(2.*pi*y - pi/2.)
    
    T = 0.5*sin(2.*pi*x)*sin(pi*y)*(1. - exp(-0.5))
    
    mean_pressure = fe.assemble(p*fe.dx)
    
    p -= mean_pressure
    
    return p, u, T
    
    
def time_verification_solution(sim):
    
    t = sim.time
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    ihat, jhat = sim.unit_vectors()
    
    u = exp(t/2)*sin(2.*pi*x)*sin(pi*y)*ihat + \
        exp(t/2)*sin(pi*x)*sin(2.*pi*y)*jhat
    
    p = -sin(pi*x - pi/2.)*sin(2.*pi*y - pi/2.)
    
    T = 0.5*sin(2.*pi*x)*sin(pi*y)*(1. - exp(-0.5*t**2))
    
    mean_pressure = fe.assemble(p*fe.dx)
    
    p -= mean_pressure
    
    return p, u, T
    
    
def dirichlet_boundary_conditions(sim, manufactured_solution):
    """Apply velocity and temperature Dirichlet BC's on every boundary.
    
    Do not apply Dirichlet BC's on the pressure.
    """
    W = sim.solution_space
    
    _, u, T = manufactured_solution
    
    return [
        fe.DirichletBC(W.sub(1), u, "on_boundary"),
        fe.DirichletBC(W.sub(2), T, "on_boundary")]
    

sim_kwargs = {
    "reynolds_number": 20.,
    "rayleigh_number": 2.5e6,
    "prandtl_number": 7.0,
    "stefan_number": 0.13,
    "density_solid_to_liquid_ratio": 0.92,
    "heat_capacity_solid_to_liquid_ratio": 0.50,
    "thermal_conductivity_solid_to_liquid_ratio": 3.8,
    "liquidus_smoothing_factor": 0.1,
    "solid_velocity_relaxation_factor": 1.e-12,
    "quadrature_degree": 4,
    }


endtime = 1.

def test__verify_second_order_spatial_convergence_via_mms(tmpdir):
    
    sim_kwargs["element_degrees"] = (1, 2, 2)
    
    sim_kwargs["timestep_size"] = endtime
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = space_verification_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        meshes = [fe.UnitSquareMesh(n, n) for n in (4, 8, 16)],
        norms = ("L2", "H1", "H1"),
        expected_orders = (None, 2, 2),
        decimal_places = 1,
        endtime = endtime)


def test__verify_second_order_temporal_convergence_via_mms(tmpdir):
    
    sim_kwargs["element_degrees"] = (2, 3, 3)
    
    sim_kwargs["mesh"] = fe.UnitSquareMesh(16, 16)
    
    sim_kwargs["time_stencil_size"] = 3
    
    sapphire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = time_verification_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        norms = (None, "L2", "L2"),
        expected_orders = (None, 2, 2),
        decimal_places = 1,
        timestep_sizes = (1/2, 1/4, 1/8, 1/16, 1/32),
        endtime = endtime)
