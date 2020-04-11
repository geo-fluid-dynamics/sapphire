"""Verify accuracy of the Laplace solver."""
import firedrake as fe 
import sapphire.mms
import sapphire.simulations.laplace as sim_module

    
def manufactured_solution(sim):
    
    sin, pi = fe.sin, fe.pi
    
    x = fe.SpatialCoordinate(sim.mesh)[0]
    
    return sin(2.*pi*x)
    
    
def test__verify_convergence_order_via_mms():
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitIntervalMesh(n) for n in (8, 16, 32)],
        norms = ("L2",),
        expected_orders = (2,),
        decimal_places = 1)
    