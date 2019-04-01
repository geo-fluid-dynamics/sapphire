import firedrake as fe 
import sunfire.mms
from sunfire.simulations import convection_coupled_phasechange as sim_module


def manufactured_solution(sim):

    pi, sin, cos, exp = fe.pi, fe.sin, fe.cos, fe.exp
    
    x = fe.SpatialCoordinate(sim.mesh)
    
    t = sim.time
    
    t_f = fe.Constant(1.)
    
    ihat, jhat = sunfire.sim.unit_vectors(sim.mesh)
    
    u = exp(t)*sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
        exp(t)*sin(pi*x[0])*sin(2.*pi*x[1])*jhat
    
    p = -sin(pi*x[0])*sin(2.*pi*x[1])
    
    T = 0.5*sin(2.*pi*x[0])*sin(pi*x[1])*(1. - exp(-t**2))
    
    return p, u, T
    
    
def test__verify__second_order_spatial_convergence__via_mms(
        sim_constructor_kwargs = {
            "quadrature_degree": 4,
            "element_degree": 1,
            "time_stencil_size": 3},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "heat_capacity_solid_to_liquid_ratio": 0.500,
            "thermal_conductivity_solid_to_liquid_ratio": 2.14/0.561,
            "smoothing": 1./16.},
        mesh_sizes = (2, 4, 8),
        timestep_size = 1./64.,
        tolerance = 0.13):
    
    rt = sim_constructor_kwargs["time_stencil_size"] - 1
    
    sunfire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        sim_constructor_kwargs = sim_constructor_kwargs,
        meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
        parameters = parameters,
        expected_order = 2,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 0.5)
        
        
def test__verify__second_order_temporal_convergence__via_mms(
        sim_constructor_kwargs = {
            "quadrature_degree": 4,
            "element_degree": 1,
            "time_stencil_size": 3},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "heat_capacity_solid_to_liquid_ratio": 0.500,
            "thermal_conductivity_solid_to_liquid_ratio": 2.14/0.561,
            "smoothing": 1./16.},
        meshsize = 24,
        timestep_sizes = (1./3., 1./9., 1./27.),
        tolerance = 0.3):
    
    rx = sim_constructor_kwargs["element_degree"] + 1
    
    mesh = fe.UnitSquareMesh(meshsize, meshsize)
    
    h = mesh.cell_sizes((0.,)*mesh.geometric_dimension())
    
    sunfire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        sim_constructor_kwargs = sim_constructor_kwargs,
        mesh = mesh,
        parameters = parameters,
        expected_order = 2,
        tolerance = tolerance,
        timestep_sizes = timestep_sizes,
        endtime = 0.5)

        
def test__verify__third_order_spatial_convergence__via_mms(
        sim_constructor_kwargs = {
            "quadrature_degree": 2,
            "element_degree": 2,
            "time_stencil_size": 4},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "heat_capacity_solid_to_liquid_ratio": 0.500,
            "thermal_conductivity_solid_to_liquid_ratio": 2.14/0.561,
            "smoothing": 1./16.},
        mesh_sizes = (3, 6, 12, 24),
        timestep_size = 1./64.,
        tolerance = 0.32):
    
    rt = sim_constructor_kwargs["time_stencil_size"] - 1
    
    sunfire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        sim_constructor_kwargs = sim_constructor_kwargs,
        meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
        parameters = parameters,
        expected_order = 3,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 0.32)
        
        
def test__verify__third_order_temporal_convergence__via_mms(
        sim_constructor_kwargs = {
            "quadrature_degree": 2,
            "element_degree": 2,
            "time_stencil_size": 4},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "heat_capacity_solid_to_liquid_ratio": 0.500,
            "thermal_conductivity_solid_to_liquid_ratio": 2.14/0.561,
            "smoothing": 1./16.},
        meshsize = 32,
        timestep_sizes = (1./2., 1./4., 1./8., 1./16.),
        tolerance = 0.3):
    
    rx = sim_constructor_kwargs["element_degree"] + 1
    
    mesh = fe.UnitSquareMesh(meshsize, meshsize)
    
    h = mesh.cell_sizes((0.,)*mesh.geometric_dimension())
    
    sunfire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        sim_constructor_kwargs = sim_constructor_kwargs,
        mesh = mesh,
        parameters = parameters,
        expected_order = 3,
        tolerance = tolerance,
        timestep_sizes = timestep_sizes,
        endtime = 0.5)
        