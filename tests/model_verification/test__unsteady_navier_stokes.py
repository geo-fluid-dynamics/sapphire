import firedrake as fe 
import sunfire.mms
from sunfire.simulation import unit_vectors
from sunfire.simulations import unsteady_navier_stokes as sim_module


def manufactured_solution(sim):
    
    exp, sin, pi = fe.exp, fe.sin, fe.pi
    
    x = fe.SpatialCoordinate(sim.mesh)
    
    t = sim.time
    
    ihat, jhat = unit_vectors(sim.mesh)
    
    u = exp(t)*(sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
        sin(pi*x[0])*sin(2.*pi*x[1])*jhat)
    
    p = -0.5*(u[0]**2 + u[1]**2)
    
    return u, p
    
    
def test__verify_spatial_convergence__second_order__via_mms(
        mesh_sizes = (3, 6, 12, 24),
        timestep_size = 1./32.,
        tolerance = 0.3):
    
    sunfire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
        sim_constructor_kwargs = {
            "quadrature_degree": 4,
            "element_degree": 1,
            "time_stencil_size": 2},
        expected_order = 2,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 1.)
    
 
def test__verify_temporal_convergence__first_order__via_mms(
        meshsize = 32,
        timestep_sizes = (1./2., 1./4., 1./8.),
        tolerance = 0.1):
    
    sunfire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        mesh = fe.UnitSquareMesh(meshsize, meshsize),
        sim_constructor_kwargs = {
            "quadrature_degree": 4,
            "element_degree": 1,
            "time_stencil_size": 2},
        expected_order = 1,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance)
    