"""Verify accuracy of the enthalpy-porosity solver

for convection-coupled phase-change.

The pressure accuracy is not verified.
"""
import sys
import pathlib
import firedrake as fe 
import sapphire.mms
import sapphire.test
from sapphire.simulations import enthalpy_porosity as sim_module


tempdir = sapphire.test.datadir


pi, sin, exp, dot = fe.pi, fe.sin, fe.exp, fe.dot


def space_verification_solution(sim):
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    ihat, jhat = sim.unit_vectors()
    
    u = exp(0.5)*sin(2.*pi*x)*sin(pi*y)*ihat + \
        exp(0.5)*sin(pi*x)*sin(2.*pi*y)*jhat
    
    # The pressure derivative is zero at x = 0, 1 and y = 0, 1
    # so that Dirichlet BC's do not have to be applied
    # and non-homogeneous Neumann BC's don't need to be handled.
    p = -sin(pi*x - pi/2.)*sin(2.*pi*y - pi/2.)
    
    T = 0.5*sin(2.*pi*x)*sin(pi*y)*(1. - exp(-0.5))
    
    return p, u, T
    
    
def time_verification_solution(sim):
    
    t = sim.time
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    ihat, jhat = sim.unit_vectors()
    
    u = exp(t/2)*sin(2.*pi*x)*sin(pi*y)*ihat + \
        exp(t/2)*sin(pi*x)*sin(2.*pi*y)*jhat
    
    # The pressure derivative is zero at x = 0, 1 and y = 0, 1
    # so that Dirichlet BC's do not have to be applied
    # and non-homogeneous Neumann BC's don't need to be handled.
    p = -sin(pi*x - pi/2.)*sin(2.*pi*y - pi/2.)
    
    T = 0.5*sin(2.*pi*x)*sin(pi*y)*(1. - exp(-0.5*t**2))
    
    return p, u, T
    

sim_kwargs = {
    "grashof_number": 3.6e5,
    "prandtl_number": 7.0,
    "stefan_number": 0.13,
    "density_solid_to_liquid_ratio": 0.92,
    "heat_capacity_solid_to_liquid_ratio": 0.50,
    "thermal_conductivity_solid_to_liquid_ratio": 3.8,
    "liquidus_smoothing_factor": 0.1,
    "solid_velocity_relaxation_factor": 1.e-12,
    "pressure_penalty_factor": 1.e-4,
    "quadrature_degree": None,
    "element_degree": None,
    }


endtime = 1.

def test__verify__second_order_spatial_convergence__via_mms(
        tempdir):
        
    sim_kwargs["element_degree"] = (1, 2, 2)
    
    sim_kwargs["timestep_size"] = endtime
    
    testdir = "{}/{}/".format(
        __name__.replace(".", "/"), sys._getframe().f_code.co_name)
    
    outdir_path = pathlib.Path(tempdir) / testdir
    
    outdir_path.mkdir(parents = True, exist_ok = True) 
    
    with open(outdir_path / "convergence.csv", "w") as outfile:
        
        sapphire.mms.verify_spatial_order_of_accuracy(
            sim_module = sim_module,
            sim_kwargs = sim_kwargs,
            manufactured_solution = space_verification_solution,
            meshes = [fe.UnitSquareMesh(n, n) for n in (4, 8, 16, 32, 64)],
            norms = (None, "H1", "H1"),
            expected_orders = (None, 2, 2),
            decimal_places = 1,
            endtime = endtime,
            outfile = outfile)


def test__verify__third_order_spatial_convergence__via_mms(
        tempdir):
    
    sim_kwargs["element_degree"] = (2, 3, 3)
    
    sim_kwargs["timestep_size"] = endtime
    
    testdir = "{}/{}/".format(
        __name__.replace(".", "/"), sys._getframe().f_code.co_name)
    
    outdir_path = pathlib.Path(tempdir) / testdir
    
    outdir_path.mkdir(parents = True, exist_ok = True) 
    
    with open(outdir_path / "convergence.csv", "w") as outfile:
        
        sapphire.mms.verify_spatial_order_of_accuracy(
            sim_module = sim_module,
            sim_kwargs = sim_kwargs,
            manufactured_solution = space_verification_solution,
            meshes = [fe.UnitSquareMesh(n, n) for n in (4, 8, 16, 32)],
            norms = (None, "H1", "H1"),
            expected_orders = (None, 3, 3),
            decimal_places = 1,
            endtime = endtime,
            outfile = outfile)


def test__verify__second_order_temporal_convergence__via_mms(
        tempdir):
    
    sim_kwargs["element_degree"] = (1, 2, 2)
    
    meshsize = 32
    
    sim_kwargs["mesh"] = fe.UnitSquareMesh(meshsize, meshsize)
    
    testdir = "{}/{}/".format(
        __name__.replace(".", "/"), sys._getframe().f_code.co_name)
    
    outdir_path = pathlib.Path(tempdir) / testdir
    
    outdir_path.mkdir(parents = True, exist_ok = True) 
    
    with open(outdir_path / "convergence.csv", "w") as outfile:
    
        sapphire.mms.verify_temporal_order_of_accuracy(
            sim_module = sim_module,
            sim_kwargs = sim_kwargs,
            manufactured_solution = time_verification_solution,
            norms = (None, "L2", "L2"),
            expected_orders = (None, 2, 2),
            decimal_places = 1,
            timestep_sizes = (1/2, 1/4, 1/8, 1/16, 1/32),
            endtime = endtime,
            outfile = outfile)
