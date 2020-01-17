import firedrake as fe 
import sapphire.mms
import sapphire.simulations.navier_stokes as sim_module


def manufactured_solution(sim):
    
    sin, pi = fe.sin, fe.pi
    
    x = fe.SpatialCoordinate(sim.mesh)
    
    ihat, jhat = sapphire.simulation.unit_vectors(sim.mesh)
    
    u = sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
        sin(pi*x[0])*sin(2.*pi*x[1])*jhat
    
    p = -0.5*(u[0]**2 + u[1]**2)
    
    return u, p
    
    
def test__verify_convergence_order_via_mms(
        mesh_sizes = (8, 16, 32), tolerance = 0.2):
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
        norms = ("H1", "L2"),
        expected_orders = (2, 2),
        tolerance = tolerance)
    