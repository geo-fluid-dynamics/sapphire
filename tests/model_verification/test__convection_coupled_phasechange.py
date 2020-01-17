import sys
import pathlib
import firedrake as fe 
import sapphire.mms
import sapphire.test
from sapphire.simulations import convection_coupled_phasechange as sim_module


tempdir = sapphire.test.datadir


pi, sin, exp, dot = fe.pi, fe.sin, fe.exp, fe.dot


def space_verification_solution(sim):
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    ihat, jhat = sapphire.simulation.unit_vectors(sim.mesh)
    
    u = sin(2.*pi*x)*sin(pi*y)*ihat + \
        sin(pi*x)*sin(2.*pi*y)*jhat
    
    p = -sin(pi*x)*sin(2.*pi*y)
    
    T = 0.5*sin(2.*pi*x)*sin(pi*y)
    
    return p, u, T
    
    
def time_verification_solution(sim):
    
    t = sim.time
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    ihat, jhat = sapphire.simulation.unit_vectors(sim.mesh)
    
    u = exp(t/2)*sin(2.*pi*x)*sin(pi*y)*ihat + \
        exp(t/2)*sin(pi*x)*sin(2.*pi*y)*jhat
    
    p = -sin(pi*x)*sin(2.*pi*y)
    
    T = 0.5*sin(2.*pi*x)*sin(pi*y)*(1. - exp(-0.5*t**2))
    
    return p, u, T
    

parameters = {
    "grashof_number": 3.6e5,
    "prandtl_number": 7.0,
    "stefan_number": 0.13,
    "density_solid_to_liquid_ratio": 0.92,
    "heat_capacity_solid_to_liquid_ratio": 0.50,
    "thermal_conductivity_solid_to_liquid_ratio": 3.8,
    "liquidus_smoothing_factor": 0.5,
    "solid_velocity_relaxation_factor": 1.e-12,
    "pressure_penalty_factor": 1.e-7,
    }
    

quadrature_degree = None

endtime = 1.

def test__verify__taylor_hood_second_order_spatial_convergence__via_mms(
        tempdir):
    
    testdir = "{}/{}/".format(
        __name__.replace(".", "/"), sys._getframe().f_code.co_name)
    
    outdir_path = pathlib.Path(tempdir) / testdir
    
    outdir_path.mkdir(parents = True, exist_ok = True) 
    
    with open(outdir_path / "convergence.csv", "w") as outfile:
        
        sapphire.mms.verify_spatial_order_of_accuracy(
            sim_module = sim_module,
            manufactured_solution = space_verification_solution,
            meshes = [fe.UnitSquareMesh(n, n) for n in (4, 8, 16, 32)],
            parameters = parameters,
            sim_constructor_kwargs = {
                "element_degree": (1, 2, 2),
                "quadrature_degree": quadrature_degree},
            norms = ("L2", "H1", "H1"),
            expected_orders = (None, 2, 2),
            tolerance = 0.1,
            timestep_size = endtime,
            endtime = endtime,
            outfile = outfile)
            

def test__verify__taylor_hood_third_order_spatial_convergence__via_mms(
        tempdir):
    
    testdir = "{}/{}/".format(
        __name__.replace(".", "/"), sys._getframe().f_code.co_name)
    
    outdir_path = pathlib.Path(tempdir) / testdir
    
    outdir_path.mkdir(parents = True, exist_ok = True) 
    
    with open(outdir_path / "convergence.csv", "w") as outfile:
        
        sapphire.mms.verify_spatial_order_of_accuracy(
            sim_module = sim_module,
            manufactured_solution = space_verification_solution,
            meshes = [fe.UnitSquareMesh(n, n) for n in (4, 8, 16, 32)],
            parameters = parameters,
            sim_constructor_kwargs = {
                "element_degree": (2, 3, 3),
                "quadrature_degree": quadrature_degree},
            norms = ("L2", "H1", "H1"),
            expected_orders = (None, 3, 3),
            tolerance = 0.1,
            timestep_size = endtime,
            endtime = endtime,
            outfile = outfile)
            

def test__verify__equal_order_first_order_spatial_convergence__via_mms(
        tempdir):
    
    testdir = "{}/{}/".format(
        __name__.replace(".", "/"), sys._getframe().f_code.co_name)
    
    outdir_path = pathlib.Path(tempdir) / testdir
    
    outdir_path.mkdir(parents = True, exist_ok = True) 
    
    with open(outdir_path / "convergence.csv", "w") as outfile:
        
        sapphire.mms.verify_spatial_order_of_accuracy(
            sim_module = sim_module,
            manufactured_solution = space_verification_solution,
            meshes = [fe.UnitSquareMesh(n, n) for n in (4, 8, 16, 32)],
            parameters = parameters,
            sim_constructor_kwargs = {
                "element_degree": (1, 1, 1),
                "quadrature_degree": quadrature_degree},
            norms = ("L2", "H1", "H1"),
            expected_orders = (None, 1, 1),
            tolerance = 0.1,
            timestep_size = endtime,
            endtime = endtime,
            outfile = outfile)


def test__verify__second_order_temporal_convergence__via_mms(
        tempdir):
    
    testdir = "{}/{}/".format(
        __name__.replace(".", "/"), sys._getframe().f_code.co_name)
    
    outdir_path = pathlib.Path(tempdir) / testdir
    
    outdir_path.mkdir(parents = True, exist_ok = True) 
    
    meshsize = 32
    
    with open(outdir_path / "convergence.csv", "w") as outfile:
    
        sapphire.mms.verify_temporal_order_of_accuracy(
            sim_module = sim_module,
            manufactured_solution = time_verification_solution,
            mesh = fe.UnitSquareMesh(meshsize, meshsize),
            parameters = parameters,
            sim_constructor_kwargs = {"element_degree": (1, 2, 2)},
            norms = ("L2", "L2", "L2"),
            expected_orders = (None, 2, 2),
            tolerance = 0.2,
            timestep_sizes = (1/2, 1/4, 1/8, 1/16),
            endtime = endtime,
            outfile = outfile)
