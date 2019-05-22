import firedrake as fe
import sapphire.mms
import sapphire.simulations.convection_diffusion as sim_module


sin, pi = fe.sin, fe.pi

def manufactured_solution(sim):
    
    x = fe.SpatialCoordinate(sim.mesh)
    
    return sin(2.*pi*x[0])*sin(pi*x[1])
    
    
def advection_velocity(mesh):

    x = fe.SpatialCoordinate(mesh)
    
    ihat, jhat = sapphire.simulation.unit_vectors(mesh)
    
    return sin(2.*pi*x[0])*sin(4.*pi*x[1])*ihat \
        + sin(pi*x[0])*sin(2.*pi*x[1])*jhat
    
    
def test__verify_convergence_order_via_mms(
        mesh_sizes = (8, 16, 32), tolerance = 0.1):
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
        sim_constructor_kwargs = {"advection_velocity": advection_velocity},
        parameters = {"diffusion_coefficient": 0.1},
        expected_order = 2,
        tolerance = tolerance)
