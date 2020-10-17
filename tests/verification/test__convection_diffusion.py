"""Verify accuracy of the convection-diffusion solver."""
import firedrake as fe
import sapphire.mms
from sapphire.simulations.convection_diffusion import Simulation


def strong_residual(sim, solution):
    
    x = fe.SpatialCoordinate(sim.mesh)
    
    u = solution
    
    a = sim.advection_velocity
    
    d = sim.diffusion_coefficient
    
    dot, grad, div = fe.dot, fe.grad, fe.div
    
    return dot(a, grad(u)) - d*div(grad(u))
    

sin, pi = fe.sin, fe.pi

def manufactured_solution(sim):
    
    x = fe.SpatialCoordinate(sim.solution.function_space().mesh())
    
    return sin(2.*pi*x[0])*sin(pi*x[1])
    
    
def advection_velocity(mesh):

    x = fe.SpatialCoordinate(mesh)
    
    ihat, jhat = sapphire.simulation.unit_vectors(mesh)
    
    return sin(2.*pi*x[0])*sin(4.*pi*x[1])*ihat \
        + sin(pi*x[0])*sin(2.*pi*x[1])*jhat
    
    
class UnitSquareUniformMeshSimulation(Simulation):
    
    def __init__(self, *args,
            meshcell_size,
            **kwargs):
        
        n = int(round(1/meshcell_size))
        
        kwargs["mesh"] = fe.UnitSquareMesh(n, n)
        
        super().__init__(*args, **kwargs)
        
    
def test__verify_convergence_order__via_mms():
    
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = "meshcell_size",
        discretization_parameter_values = [1/n for n in (8, 16, 32)],
        Simulation = UnitSquareUniformMeshSimulation,
        sim_kwargs = {
            "diffusion_coefficient": 0.1,
            "advection_velocity": advection_velocity,
            },
        strong_residual = strong_residual,
        manufactured_solution = manufactured_solution,
        norms = ("H1",),
        expected_orders = (1,),
        decimal_places = 1)
