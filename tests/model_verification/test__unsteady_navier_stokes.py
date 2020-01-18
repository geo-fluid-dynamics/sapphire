import firedrake as fe 
import sapphire.mms
from sapphire.simulation import unit_vectors
from sapphire.simulations import unsteady_navier_stokes as sim_module


def manufactured_solution(sim):
    
    exp, sin, pi = fe.exp, fe.sin, fe.pi
    
    x = fe.SpatialCoordinate(sim.mesh)
    
    t = sim.time
    
    ihat, jhat = unit_vectors(sim.mesh)
    
    u = exp(t)*(sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
        sin(pi*x[0])*sin(2.*pi*x[1])*jhat)
    
    p = -0.5*(u[0]**2 + u[1]**2)
    
    return u, p
    
    
parameters = {
    "quadrature_degree": 4,
    "element_degree": 1,
    "time_stencil_size": 2}
    
def test__verify_spatial_convergence__second_order__via_mms():
    
    parameters["timestep_size"] = 1./32.
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_parameters = parameters,
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitSquareMesh(n, n) for n in (3, 6, 12, 24)],
        norms = ("H1", "L2"),
        expected_orders = (2, 2),
        tolerance = 0.3,
        endtime = 1.)
    
 
def test__verify_temporal_convergence__first_order__via_mms():
    
    meshsize = 32
    
    parameters["mesh"] = fe.UnitSquareMesh(meshsize, meshsize)
    
    sapphire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        sim_parameters = parameters,
        manufactured_solution = manufactured_solution,
        norms = ("L2", "L2"),
        expected_orders = (None, 1),
        endtime = 1.,
        timestep_sizes = (1./2., 1./4., 1./8.),
        tolerance = 0.1)
    