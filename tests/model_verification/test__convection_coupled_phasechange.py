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
    
    
Gr = 3.6e5

Pr = 7.0

Ste = 0.13

rhos_over_rhol = 0.92

cs_over_cl = 0.50

kappas_over_kappal = 3.8


sigma = 0.5

tau = 1.e-12

gamma = 1.e-7

q = None


def test__verify__taylor_hood_second_order_spatial_convergence__via_mms(
        tempdir,
        parameters = {
            "grashof_number": Gr,
            "prandtl_number": Pr,
            "stefan_number": Ste,
            "density_solid_to_liquid_ratio": rhos_over_rhol,
            "heat_capacity_solid_to_liquid_ratio": cs_over_cl,
            "thermal_conductivity_solid_to_liquid_ratio": kappas_over_kappal,
            "liquidus_smoothing_factor": sigma,
            "pressure_penalty_factor": gamma,
            "solid_velocity_relaxation_factor": tau},
        sim_constructor_kwargs = {
            "element_degree": (1, 2, 2),
            "quadrature_degree": q},
        endtime = 1.,
        mesh_sizes = (4, 8, 16, 32),
        tolerance = 0.1):
    
    testdir = "{}/{}/".format(
        __name__.replace(".", "/"), sys._getframe().f_code.co_name)
    
    outdir_path = pathlib.Path(tempdir) / testdir
    
    outdir_path.mkdir(parents = True, exist_ok = True) 
    
    with open(outdir_path / "convergence.csv", "w") as outfile:
        
        sapphire.mms.verify_spatial_order_of_accuracy(
            sim_module = sim_module,
            manufactured_solution = space_verification_solution,
            meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
            parameters = parameters,
            sim_constructor_kwargs = sim_constructor_kwargs,
            norms = ("L2", "H1", "H1"),
            expected_orders = (None, 2, 2),
            tolerance = tolerance,
            timestep_size = endtime,
            endtime = endtime,
            outfile = outfile)
            

def test__verify__taylor_hood_third_order_spatial_convergence__via_mms(
        tempdir,
        parameters = {
            "grashof_number": Gr,
            "prandtl_number": Pr,
            "stefan_number": Ste,
            "density_solid_to_liquid_ratio": rhos_over_rhol,
            "heat_capacity_solid_to_liquid_ratio": cs_over_cl,
            "thermal_conductivity_solid_to_liquid_ratio": kappas_over_kappal,
            "liquidus_smoothing_factor": sigma,
            "pressure_penalty_factor": gamma,
            "solid_velocity_relaxation_factor": tau},
        sim_constructor_kwargs = {
            "element_degree": (2, 3, 3),
            "quadrature_degree": q},
        endtime = 1.,
        mesh_sizes = (4, 8, 16, 32),
        tolerance = 0.1):
    
    testdir = "{}/{}/".format(
        __name__.replace(".", "/"), sys._getframe().f_code.co_name)
    
    outdir_path = pathlib.Path(tempdir) / testdir
    
    outdir_path.mkdir(parents = True, exist_ok = True) 
    
    with open(outdir_path / "convergence.csv", "w") as outfile:
        
        sapphire.mms.verify_spatial_order_of_accuracy(
            sim_module = sim_module,
            manufactured_solution = space_verification_solution,
            meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
            parameters = parameters,
            sim_constructor_kwargs = sim_constructor_kwargs,
            norms = ("L2", "H1", "H1"),
            expected_orders = (None, 3, 3),
            tolerance = tolerance,
            timestep_size = endtime,
            endtime = endtime,
            outfile = outfile)
            

def test__verify__equal_order_first_order_spatial_convergence__via_mms(
        tempdir,
        parameters = {
            "grashof_number": Gr,
            "prandtl_number": Pr,
            "stefan_number": Ste,
            "density_solid_to_liquid_ratio": rhos_over_rhol,
            "heat_capacity_solid_to_liquid_ratio": cs_over_cl,
            "thermal_conductivity_solid_to_liquid_ratio": kappas_over_kappal,
            "liquidus_smoothing_factor": sigma,
            "pressure_penalty_factor": gamma,
            "solid_velocity_relaxation_factor": tau},
        sim_constructor_kwargs = {
            "element_degree": (1, 1, 1),
            "quadrature_degree": q},
        endtime = 1.,
        mesh_sizes = (4, 8, 16, 32),
        tolerance = 0.1):
    
    testdir = "{}/{}/".format(
        __name__.replace(".", "/"), sys._getframe().f_code.co_name)
    
    outdir_path = pathlib.Path(tempdir) / testdir
    
    outdir_path.mkdir(parents = True, exist_ok = True) 
    
    with open(outdir_path / "convergence.csv", "w") as outfile:
        
        sapphire.mms.verify_spatial_order_of_accuracy(
            sim_module = sim_module,
            manufactured_solution = space_verification_solution,
            meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
            parameters = parameters,
            sim_constructor_kwargs = sim_constructor_kwargs,
            norms = ("L2", "H1", "H1"),
            expected_orders = (None, 1, 1),
            tolerance = tolerance,
            timestep_size = endtime,
            endtime = endtime,
            outfile = outfile)


def test__verify__second_order_temporal_convergence__via_mms(
        tempdir,
        parameters = {
            "grashof_number": Gr,
            "prandtl_number": Pr,
            "stefan_number": Ste,
            "density_solid_to_liquid_ratio": rhos_over_rhol,
            "heat_capacity_solid_to_liquid_ratio": cs_over_cl,
            "thermal_conductivity_solid_to_liquid_ratio": kappas_over_kappal,
            "liquidus_smoothing_factor": sigma,
            "pressure_penalty_factor": gamma},
        sim_constructor_kwargs = {"element_degree": (1, 2, 2)},
        meshsize = 32,
        timestep_sizes = (1/2, 1/4, 1/8, 1/16),
        endtime = 1.,
        tolerance = 0.2):
    
    testdir = "{}/{}/".format(
        __name__.replace(".", "/"), sys._getframe().f_code.co_name)
    
    outdir_path = pathlib.Path(tempdir) / testdir
    
    outdir_path.mkdir(parents = True, exist_ok = True) 
    
    with open(outdir_path / "convergence.csv", "w") as outfile:
    
        sapphire.mms.verify_temporal_order_of_accuracy(
            sim_module = sim_module,
            manufactured_solution = time_verification_solution,
            mesh = fe.UnitSquareMesh(meshsize, meshsize),
            parameters = parameters,
            sim_constructor_kwargs = sim_constructor_kwargs,
            norms = ("L2", "L2", "L2"),
            expected_orders = (None, 2, 2),
            tolerance = tolerance,
            timestep_sizes = timestep_sizes,
            endtime = endtime,
            outfile = outfile)
