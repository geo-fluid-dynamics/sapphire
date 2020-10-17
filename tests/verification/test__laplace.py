"""Verify accuracy of the Laplace solver."""
import firedrake as fe 
import sapphire.mms
from sapphire.simulations.laplace import Simulation


def strong_residual(sim, solution):
    
    div, grad, = fe.div, fe.grad
    
    u = solution
    
    return div(grad(u))
    
    
def manufactured_solution(sim):
    
    sin, pi = fe.sin, fe.pi
    
    x = fe.SpatialCoordinate(sim.mesh)[0]
    
    return sin(2.*pi*x)
    
    
class UnitIntervalUniformMeshSimulation(Simulation):
    
    def __init__(self, *args,
            meshcell_size,
            **kwargs):
        
        n = int(round(1/meshcell_size))
        
        kwargs['mesh'] = fe.UnitIntervalMesh(n)
        
        super().__init__(*args, **kwargs)
        
    
def test__verify_convergence_order_via_mms():
    
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = 'meshcell_size',
        discretization_parameter_values = [1/n for n in (8, 16, 32)],
        Simulation = UnitIntervalUniformMeshSimulation,
        strong_residual = strong_residual,
        manufactured_solution = manufactured_solution,
        norms = ('L2',),
        expected_orders = (2,),
        decimal_places = 1)
    