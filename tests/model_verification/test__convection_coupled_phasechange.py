import sys
import pathlib
import firedrake as fe 
import sapphire.mms
import sapphire.test
from sapphire.simulations import convection_coupled_phasechange as sim_module


tempdir = sapphire.test.datadir

pi, sin, exp, dot = fe.pi, fe.sin, fe.exp, fe.dot

def manufactured_solution(mesh, time):

    x = fe.SpatialCoordinate(mesh)
    
    ihat, jhat = sapphire.simulation.unit_vectors(mesh)
    
    t = time
    
    u = exp(t/2)*sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
        exp(t/2)*sin(pi*x[0])*sin(2.*pi*x[1])*jhat
    
    p = -sin(pi*x[0])*sin(2.*pi*x[1])
    
    T = 0.5*sin(2.*pi*x[0])*sin(pi*x[1])*(1. - exp(-0.5*t**2))
    
    return p, u, T


def space_verification_solution(sim):

    return manufactured_solution(mesh = sim.mesh, time = 1.)

    
def time_verification_solution(sim):

    return manufactured_solution(mesh = sim.mesh, time = sim.time)

    
Gr = 3.6e5

Pr = 7.0

Ste = 0.13

rhos_over_rhol = 0.92

cs_over_cl = 0.50

kappas_over_kappal = 3.8

sigma = 0.1

endtime = 1.

def test__verify__second_order_spatial_convergence__via_mms(
        tempdir,
        parameters = {
            "grashof_number": Gr,
            "prandtl_number": Pr,
            "stefan_number": Ste,
            "density_solid_to_liquid_ratio": rhos_over_rhol,
            "heat_capacity_solid_to_liquid_ratio": cs_over_cl,
            "thermal_conductivity_solid_to_liquid_ratio": kappas_over_kappal,
            "smoothing": sigma},
        mesh_sizes = (8, 16, 32, 64),
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
            expected_order = 2,
            tolerance = tolerance,
            timestep_size = endtime,
            endtime = endtime,
            outfile = outfile)
            
            
def test__verify__second_order_spatial_convergence__low_quadrature__via_mms(
        tempdir,
        sim_constructor_kwargs = {"quadrature_degree": 2},
        parameters = {
            "grashof_number": Gr,
            "prandtl_number": Pr,
            "stefan_number": Ste,
            "density_solid_to_liquid_ratio": rhos_over_rhol,
            "heat_capacity_solid_to_liquid_ratio": cs_over_cl,
            "thermal_conductivity_solid_to_liquid_ratio": kappas_over_kappal,
            "smoothing": sigma},
        mesh_sizes = (8, 16, 32, 64),
        tolerance = 0.1):
    
    testdir = "{}/{}/".format(
        __name__.replace(".", "/"), sys._getframe().f_code.co_name)
    
    outdir_path = pathlib.Path(tempdir) / testdir
    
    outdir_path.mkdir(parents = True, exist_ok = True) 
    
    with open(outdir_path / "convergence.csv", "w") as outfile:
        
        sapphire.mms.verify_spatial_order_of_accuracy(
            sim_module = sim_module,
            sim_constructor_kwargs = sim_constructor_kwargs,
            manufactured_solution = space_verification_solution,
            meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
            parameters = parameters,
            expected_order = 2,
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
            "smoothing": sigma},
        meshsize = 48,
        timestep_sizes = (1, 1/2, 1/4, 1/8, 1/16),
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
            expected_order = 2,
            tolerance = tolerance,
            timestep_sizes = timestep_sizes,
            endtime = endtime,
            outfile = outfile)


def test__verify__second_order_temporal_convergence__low_quadrature__via_mms(
        tempdir,
        sim_constructor_kwargs = {"quadrature_degree": 2},
        parameters = {
            "grashof_number": Gr,
            "prandtl_number": Pr,
            "stefan_number": Ste,
            "density_solid_to_liquid_ratio": rhos_over_rhol,
            "heat_capacity_solid_to_liquid_ratio": cs_over_cl,
            "thermal_conductivity_solid_to_liquid_ratio": kappas_over_kappal,
            "smoothing": sigma},
        meshsize = 48,
        timestep_sizes = (1, 1/2, 1/4, 1/8, 1/16),
        tolerance = 0.2):
    
    testdir = "{}/{}/".format(
        __name__.replace(".", "/"), sys._getframe().f_code.co_name)
    
    outdir_path = pathlib.Path(tempdir) / testdir
    
    outdir_path.mkdir(parents = True, exist_ok = True) 
    
    with open(outdir_path / "convergence.csv", "w") as outfile:
    
        sapphire.mms.verify_temporal_order_of_accuracy(
            sim_module = sim_module,
            sim_constructor_kwargs = sim_constructor_kwargs,
            manufactured_solution = time_verification_solution,
            mesh = fe.UnitSquareMesh(meshsize, meshsize),
            parameters = parameters,
            expected_order = 2,
            tolerance = tolerance,
            timestep_sizes = timestep_sizes,
            endtime = endtime,
            outfile = outfile)
            