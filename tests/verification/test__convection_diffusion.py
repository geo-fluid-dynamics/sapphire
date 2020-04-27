"""Verify accuracy of the convection-diffusion solver."""
import firedrake as fe
import sapphire.mms
import sapphire.simulations.convection_diffusion as sim_module


sin, pi = fe.sin, fe.pi

def manufactured_solution(sim):
    
    x = fe.SpatialCoordinate(sim.solution.function_space().mesh())
    
    return sin(2.*pi*x[0])*sin(pi*x[1])
    
    
def advection_velocity(mesh):

    x = fe.SpatialCoordinate(mesh)
    
    ihat, jhat = sapphire.simulation.unit_vectors(mesh)
    
    return sin(2.*pi*x[0])*sin(4.*pi*x[1])*ihat \
        + sin(pi*x[0])*sin(2.*pi*x[1])*jhat
    
    
def test__verify_convergence_order__via_mms():
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = {
            "diffusion_coefficient": 0.1,
            "advection_velocity": advection_velocity,
            },
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitSquareMesh(n, n) for n in (8, 16, 32)],
        norms = ("H1",),
        expected_orders = (1,),
        decimal_places = 1)
