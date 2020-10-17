"""Verify accuracy of the enthalpy phase-change solver."""
import firedrake as fe 
import sapphire.mms
from sapphire.simulations.enthalpy import Simulation

    
def strong_residual(sim, solution):
    
    T = solution
    
    t = sim.time
    
    Ste = sim.stefan_number
    
    phil = sim.liquid_volume_fraction(temperature = T)
    
    diff, div, grad = fe.diff, fe.div, fe.grad
    
    return diff(T, t) - div(grad(T)) + 1./Ste*diff(phil, t)
    
    
def manufactured_solution(sim):
    
    x = fe.SpatialCoordinate(sim.solution.function_space().mesh())[0]
    
    t = sim.time
    
    sin, pi, exp = fe.sin, fe.pi, fe.exp
    
    return 0.5*sin(2.*pi*x)*(1. - 2*exp(-3.*t**2))


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
        discretization_parameter_values = [1/n for n in (4, 8, 16, 32)],
        Simulation = UnitIntervalUniformMeshSimulation,
        sim_kwargs = {
            "stefan_number": 0.1,
            "liquidus_smoothing_factor": 1./32.,
            "timestep_size": 1./256.,
            },
        strong_residual = strong_residual,
        manufactured_solution = manufactured_solution,
        norms = ("H1",),
        expected_orders = (1,),
        endtime = 1.,
        decimal_places = 1)
        
     
def test__verify_spatial_convergence__second_order__via_mms():
    
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = "meshcell_size",
        discretization_parameter_values = [1/n for n in (4, 8, 16)],
        Simulation = UnitIntervalUniformMeshSimulation,
        sim_kwargs = {
            "stefan_number": 0.1,
            "liquidus_smoothing_factor": 1/32,
            "element_degree": 2,
            "timestep_size": 1/256,
            },
        strong_residual = strong_residual,
        manufactured_solution = manufactured_solution,
        norms = ("H1",),
        expected_orders = (2,),
        endtime = 1.,
        decimal_places = 1)
        
        
def test__verify_temporal_convergence__first_order__via_mms():
    
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = "timestep_size",
        discretization_parameter_values = (1/16, 1/32, 1/64, 1/128),
        Simulation = UnitIntervalUniformMeshSimulation,
        sim_kwargs = {
            "stefan_number": 0.1,
            "liquidus_smoothing_factor": 1./32.,
            "meshcell_size": 1/512,
            },
        strong_residual = strong_residual,
        manufactured_solution = manufactured_solution,
        norms = ("L2",),
        expected_orders = (1,),
        endtime = 1.,
        decimal_places = 1)

    
def test__verify_temporal_convergence__second_order__via_mms():
    
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = "timestep_size",
        discretization_parameter_values = (1/32, 1/64, 1/128),
        Simulation = UnitIntervalUniformMeshSimulation,
        sim_kwargs = {
            "stefan_number": 0.1,
            "liquidus_smoothing_factor": 1/32,
            "element_degree": 2,
            "time_stencil_size": 3,
            "meshcell_size": 1/256,
            },
        strong_residual = strong_residual,
        manufactured_solution = manufactured_solution,
        norms = ("L2",),
        expected_orders = (2,),
        endtime = 1.,
        decimal_places = 1)
    