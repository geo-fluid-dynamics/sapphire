import firedrake as fe 
import sapphire.mms
from sapphire.simulations import phasechange as sim_module


def manufactured_solution(sim):
    
    x = fe.SpatialCoordinate(sim.mesh)[0]
    
    t = sim.time
    
    sin, pi, exp = fe.sin, fe.pi, fe.exp
    
    return 0.5*sin(2.*pi*x)*(1. - 2*exp(-3.*t**2))

    
def test__verify_spatial_convergence__first_order__via_mms(
        mesh_sizes = (4, 8, 16, 32),
        timestep_size = 1./256.,
        tolerance = 0.1):
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitIntervalMesh(size) for size in mesh_sizes],
        parameters = {
            "stefan_number": 0.1,
            "liquidus_smoothing_factor": 1./32.,
            },
        norms = ("H1",),
        expected_orders = (1,),
        timestep_size = timestep_size,
        endtime = 1.,
        tolerance = tolerance)
        
     
def test__verify_spatial_convergence__second_order__via_mms(
        mesh_sizes = (4, 8, 16, 32),
        timestep_size = 1./256.,
        tolerance = 0.1):
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitIntervalMesh(size) for size in mesh_sizes],
        parameters = {
            "stefan_number": 0.1,
            "liquidus_smoothing_factor": 1./32.,
            "element_degree": 2,
            },
        norms = ("H1",),
        expected_orders = (2,),
        timestep_size = timestep_size,
        endtime = 1.,
        tolerance = tolerance)
        
        
def test__verify_temporal_convergence__first_order__via_mms(
        meshsize = 256,
        timestep_sizes = (1./16., 1./32., 1./64., 1./128.),
        tolerance = 0.1):
    
    sapphire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        mesh = fe.UnitIntervalMesh(meshsize),
        parameters = {
            "stefan_number": 0.1,
            "liquidus_smoothing_factor": 1./32.,
            },
        norms = ("L2",),
        expected_orders = (1,),
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance)

    
def test__verify_temporal_convergence__second_order__via_mms(
        meshsize = 128,
        timestep_sizes = (1./64., 1./128., 1./256.),
        tolerance = 0.1):
    
    sapphire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        mesh = fe.UnitIntervalMesh(meshsize),
        parameters = {
            "stefan_number": 0.1,
            "liquidus_smoothing_factor": 1./32.,
            "element_degree": 2,
            "time_stencil_size": 3,
            },
        norms = ("L2",),
        expected_orders = (2,),
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance)
    