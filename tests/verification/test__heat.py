"""Verify accuracy of the heat equation solver."""
import firedrake as fe 
import sapphire.mms
from sapphire.simulations.heat import Simulation


def strong_residual(sim, solution):
    
    u = solution
    
    t = sim.time
    
    diff, div, grad = fe.diff, fe.div, fe.grad
    
    return diff(u, t) - div(grad(u))
    

def manufactured_solution(sim):
    
    x = fe.SpatialCoordinate(sim.mesh)[0]
    
    t = sim.time
    
    sin, pi, exp = fe.sin, fe.pi, fe.exp
    
    return sin(2.*pi*x)*exp(-pow(t, 2))
    

class UnitIntervalUniformMeshSimulation(Simulation):
    
    def __init__(self, *args,
            meshcell_size,
            **kwargs):
        
        n = int(round(1/meshcell_size))
        
        kwargs["mesh"] = fe.UnitIntervalMesh(n)
        
        super().__init__(*args, **kwargs)
        
        
def test__verify_spatial_convergence__first_order__via_mms():
    
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = "meshcell_size",
        discretization_parameter_values = [1/n for n in (4, 8, 16)],
        Simulation = UnitIntervalUniformMeshSimulation,
        sim_kwargs = {"timestep_size": 1./64.},
        strong_residual = strong_residual,
        manufactured_solution = manufactured_solution,
        norms = ("H1",),
        expected_orders = (1,),
        decimal_places = 1,
        endtime = 1.)
        
        
def test__verify_temporal_convergence__first_order__via_mms():
    
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = "timestep_size",
        discretization_parameter_values = (1/4, 1/8, 1/16, 1/32),
        Simulation = UnitIntervalUniformMeshSimulation,
        sim_kwargs = {"meshcell_size": 1/256},
        strong_residual = strong_residual,
        manufactured_solution = manufactured_solution,
        norms = ("L2",),
        expected_orders = (1,),
        endtime = 1.,
        decimal_places = 1)
    
    
def test__verify_temporal_convergence__second_order__via_mms():
    
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = "timestep_size",
        discretization_parameter_values = (1/16, 1/32, 1/64, 1/128),
        Simulation = UnitIntervalUniformMeshSimulation,
        sim_kwargs = {
            "meshcell_size": 1/128,
            "element_degree": 2,
            "time_stencil_size": 3},
        strong_residual = strong_residual,
        manufactured_solution = manufactured_solution,
        norms = ("L2",),
        expected_orders = (2,),
        endtime = 1.,
        decimal_places = 1)
        
        
def test__verify_temporal_convergence__third_order__via_mms():
    
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = "timestep_size",
        discretization_parameter_values =  (1/4, 1/8, 1/16, 1/32),
        Simulation = UnitIntervalUniformMeshSimulation,
        sim_kwargs = {
            "meshcell_size": 1/256,
            "element_degree": 2,
            "time_stencil_size": 4},
        strong_residual = strong_residual,
        manufactured_solution = manufactured_solution,
        norms = ("L2",),
        expected_orders = (3,),
        endtime = 1.,
        decimal_places = 1)
        