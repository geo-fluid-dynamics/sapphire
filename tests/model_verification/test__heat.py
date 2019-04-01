import firedrake as fe 
import sunfire.mms
import sunfire.simulations.heat as sim_module


def manufactured_solution(sim):
    
    x = fe.SpatialCoordinate(sim.mesh)[0]
    
    t = sim.time
    
    sin, pi, exp = fe.sin, fe.pi, fe.exp
    
    return sin(2.*pi*x)*exp(-pow(t, 2))
    
    
def test__verify_spatial_convergence__second_order__via_mms(
        mesh_sizes = (4, 8, 16, 32),
        timestep_size = 1./64.,
        tolerance = 0.1):
    
    sunfire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitIntervalMesh(n) for n in mesh_sizes],
        sim_constructor_kwargs = {
            "element_degree": 1,
            "time_stencil_size": 2},
        expected_order = 2,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 1.)
        
        
def test__verify_temporal_convergence__first_order__via_mms(
        meshsize = 256,
        timestep_sizes = (1./4., 1./8., 1./16., 1./32.),
        tolerance = 0.1):
    
    sunfire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        mesh = fe.UnitIntervalMesh(meshsize),
        sim_constructor_kwargs = {
            "element_degree": 1,
            "time_stencil_size": 2},
        expected_order = 1,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance)
    
    
def test__verify_temporal_convergence__second_order__via_mms(
        meshsize = 128,
        timestep_sizes = (
            1./4., 1./8., 1./16., 1./32., 1./64., 1./128.),
        tolerance = 0.1):
    
    sunfire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        mesh = fe.UnitIntervalMesh(meshsize),
        sim_constructor_kwargs = {
            "element_degree": 2,
            "time_stencil_size": 3},
        expected_order = 2,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance)
        
        
def test__verify_temporal_convergence__third_order__via_mms(
        meshsize = 128,
        timestep_sizes = (1./4., 1./8., 1./16., 1./32.),
        tolerance = 0.1):
    
    sunfire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution,
        mesh = fe.UnitIntervalMesh(meshsize),
        sim_constructor_kwargs = {
            "element_degree": 2,
            "time_stencil_size": 4},
        expected_order = 3,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance)
        