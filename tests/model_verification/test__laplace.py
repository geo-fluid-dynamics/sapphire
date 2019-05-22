import firedrake as fe 
import sapphire.mms
import sapphire.simulations.laplace as sim_module

    
def manufactured_solution(sim):
    
    sin, pi = fe.sin, fe.pi
    
    x = fe.SpatialCoordinate(sim.mesh)[0]
    
    return sin(2.*pi*x)
    
    
def test__verify_convergence_order_via_mms(
        mesh_sizes = (8, 16, 32), tolerance = 0.1):
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitIntervalMesh(n) for n in mesh_sizes],
        expected_order = 2,
        tolerance = tolerance)
    