import firedrake as fe 
import sapphire.mms
from sapphire.simulation import unit_vectors
from sapphire.simulations import \
    unsteady_navier_stokes_boussinesq as sim_module


def space_verification_solution(sim):
    
    sin, pi = fe.sin, fe.pi
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    u0 = sin(2.*pi*x)*sin(pi*y)
    
    u1 = sin(pi*x)*sin(2.*pi*y)
    
    ihat, jhat = sim.unit_vectors()
    
    u = (u0*ihat + u1*jhat)
    
    p = -0.5*sin(pi*x)*sin(pi*y)
    
    T = sin(2.*pi*x)*sin(pi*y)
    
    return p, u, T
    

def time_verification_solution(sim):
    
    exp, sin, pi = fe.exp, fe.sin, fe.pi
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    t = sim.time
    
    u0 = sin(2.*pi*x)*sin(pi*y)
    
    u1 = sin(pi*x)*sin(2.*pi*y)
    
    ihat, jhat = sim.unit_vectors()
    
    u = exp(t)*(u0*ihat + u1*jhat)
    
    p = -0.5*sin(pi*x)*sin(pi*y)
    
    T = exp(t)*sin(2.*pi*x)*sin(pi*y)
    
    return p, u, T
    

sim_kwargs = {
    "grashof_number": 2.,
    "prandtl_number": 0.71,
    "quadrature_degree": None}
    
def test__verify_spatial_convergence__second_order__via_mms():
    
    sim_kwargs["timestep_size"] = 1.
    
    sim_kwargs["element_degree"] = (1, 2, 2)
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = space_verification_solution,
        endtime = 1.,
        meshes = [fe.UnitSquareMesh(n, n) for n in (4, 8, 16, 32, 64)],
        norms = ("L2", "H1", "H1"),
        expected_orders = (2, 2, 2),
        tolerance = 0.1)
    
 
def test__verify_temporal_convergence__first_order__via_mms():
    
    meshsize = 48
    
    sim_kwargs["mesh"] = fe.UnitSquareMesh(meshsize, meshsize)
    
    sim_kwargs["element_degree"] = (2, 3, 3)
    
    sim_kwargs["time_stencil_size"] = 2
    
    sapphire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = time_verification_solution,
        endtime = 1.,
        timestep_sizes = (1./2., 1./4., 1./8.),
        norms = (None, "L2", "L2"),
        expected_orders = (None, 1, 1),
        tolerance = 0.1)
    