import sys
import pathlib
import firedrake as fe 
import sapphire.mms
import sapphire.test
from sapphire.simulations import convection_coupled_phasechange as sim_module


tempdir = sapphire.test.datadir

def manufactured_solution(sim):

    pi, sin, cos, exp = fe.pi, fe.sin, fe.cos, fe.exp
    
    x = fe.SpatialCoordinate(sim.mesh)
    
    t = sim.time
    
    t_f = fe.Constant(1.)
    
    ihat, jhat = sapphire.simulation.unit_vectors(sim.mesh)
    
    u = exp(t)*sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
        exp(t)*sin(pi*x[0])*sin(2.*pi*x[1])*jhat
    
    p = -sin(pi*x[0])*sin(2.*pi*x[1])
    
    T = 0.5*sin(2.*pi*x[0])*sin(pi*x[1])*(1. - exp(-t**2))
    
    return p, u, T
    

Gr = 3.6e5

Pr = 5.

Ste = 0.2

rhos_over_rhol = 0.92

cs_over_cl = 0.50

kappas_over_kappal = 3.8

sigma = 1/16

endtime = 0.5

q = None

def test__verify__second_order_spatial_convergence__via_mms(
        tempdir,
        sim_constructor_kwargs = {
            "quadrature_degree": q,
            "element_degree": 1,
            "time_stencil_size": 3},
        parameters = {
            "grashof_number": Gr,
            "prandtl_number": Pr,
            "stefan_number": Ste,
            "density_solid_to_liquid_ratio": rhos_over_rhol,
            "heat_capacity_solid_to_liquid_ratio": cs_over_cl,
            "thermal_conductivity_solid_to_liquid_ratio": kappas_over_kappal,
            "smoothing": sigma},
        mesh_sizes = (2, 4, 8),
        timestep_size = 1./64.,
        tolerance = 0.2):
    
    rt = sim_constructor_kwargs["time_stencil_size"] - 1
    
    outpath = tempdir.join("{}/{}/".format(
        __loader__.fullname,  sys._getframe().f_code.co_name))
    
    outpath.mkdir(parents = True, exist_ok = True)
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        sim_constructor_kwargs = sim_constructor_kwargs,
        meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
        parameters = parameters,
        expected_order = 2,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = endtime,
        outfile = open(outpath.join("convergence_rt{}_Deltat{}.csv".format(
            rt, timestep_size))))
        
        
def test__verify__second_order_temporal_convergence__via_mms(
        tempdir,
        sim_constructor_kwargs = {
            "quadrature_degree": q,
            "element_degree": 1,
            "time_stencil_size": 3},
        parameters = {
            "grashof_number": Gr,
            "prandtl_number": Pr,
            "stefan_number": Ste,
            "density_solid_to_liquid_ratio": rhos_over_rhol,
            "heat_capacity_solid_to_liquid_ratio": cs_over_cl,
            "thermal_conductivity_solid_to_liquid_ratio": kappas_over_kappal,
            "smoothing": sigma},
        meshsize = 24,
        timestep_sizes = (1./3., 1./9., 1./27.),
        tolerance = 0.3):
    
    rx = sim_constructor_kwargs["element_degree"] + 1
    
    mesh = fe.UnitSquareMesh(meshsize, meshsize)
    
    h = mesh.cell_sizes((0.,)*mesh.geometric_dimension())
    
    sapphire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        sim_constructor_kwargs = sim_constructor_kwargs,
        mesh = mesh,
        parameters = parameters,
        expected_order = 2,
        tolerance = tolerance,
        timestep_sizes = timestep_sizes,
        endtime = endtime,
        outfile = open(tempdir.join("{}/rt2_rx{}_h{}.csv".format(
            outdir, rx, h))))

        
def test__verify__third_order_spatial_convergence__via_mms(
        tempdir,
        sim_constructor_kwargs = {
            "quadrature_degree": q,
            "element_degree": 2,
            "time_stencil_size": 4},
        parameters = {
            "grashof_number": Gr,
            "prandtl_number": Pr,
            "stefan_number": Ste,
            "density_solid_to_liquid_ratio": rhos_over_rhol,
            "heat_capacity_solid_to_liquid_ratio": cs_over_cl,
            "thermal_conductivity_solid_to_liquid_ratio": kappas_over_kappal,
            "smoothing": sigma},
        mesh_sizes = (4, 8, 16, 32),
        timestep_size = 1/64,
        tolerance = 0.2):
    
    tempdir_path = pathlib.Path(tempdir)
    
    testdir = "{}/{}/".format(
        __name__.replace(".", "/"), sys._getframe().f_code.co_name)
    
    outdir_path = tempdir_path / testdir
    
    outdir_path.mkdir(parents = True, exist_ok = True) 
    
    rt = sim_constructor_kwargs["time_stencil_size"] - 1
    
    filename = "rt{}_Deltat{}.csv".format(rt, timestep_size)
    
    outfile_path = outdir_path / filename
    
    with open(outfile_path, "w") as outfile:
        
        sapphire.mms.verify_spatial_order_of_accuracy(
            sim_module = sim_module,
            manufactured_solution = manufactured_solution,
            sim_constructor_kwargs = sim_constructor_kwargs,
            meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
            parameters = parameters,
            expected_order = 3,
            tolerance = tolerance,
            timestep_size = timestep_size,
            endtime = endtime,
            outfile = outfile)
            
        
def test__verify__third_order_temporal_convergence__via_mms(
        tempdir,
        sim_constructor_kwargs = {
            "quadrature_degree": q,
            "element_degree": 2,
            "time_stencil_size": 4},
        parameters = {
            "grashof_number": Gr,
            "prandtl_number": Pr,
            "stefan_number": Ste,
            "density_solid_to_liquid_ratio": rhos_over_rhol,
            "heat_capacity_solid_to_liquid_ratio": cs_over_cl,
            "thermal_conductivity_solid_to_liquid_ratio": kappas_over_kappal,
            "smoothing": sigma},
        meshsize = 24,
        timestep_sizes = (1./4., 1./8., 1./16.),
        tolerance = 0.3):
    
    rx = sim_constructor_kwargs["element_degree"] + 1
    
    mesh = fe.UnitSquareMesh(meshsize, meshsize)
    
    h = mesh.cell_sizes((0.,)*mesh.geometric_dimension())
    
    sapphire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        sim_constructor_kwargs = sim_constructor_kwargs,
        mesh = mesh,
        parameters = parameters,
        expected_order = 3,
        tolerance = tolerance,
        timestep_sizes = timestep_sizes,
        endtime = endtime,
        outfile = open(tempdir.join("{}/rt3_rx{}_h{}.csv".format(
            outdir, rx, h))))
        