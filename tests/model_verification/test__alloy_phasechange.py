import firedrake as fe 
import sapphire.mms
from sapphire.simulations import alloy_phasechange as sim_module


T_m = 1.

Ste = 0.4

Le = 8.

c_sl = 0.5

k_sl = 3.

q = 2

endtime = 0.0001

def manufactured_solution(sim, time):
    
    hmin = 0.
    
    hmax = 1. - T_m + 1./Ste
    
    Smin = 0.
    
    Smax = 1.
    
    x = fe.SpatialCoordinate(sim.mesh)[0]
    
    t = time
    
    cos, sin, pi = fe.cos, fe.sin, fe.pi
    
    h = hmin + 0.5*(hmax - hmin)*(1 + sin(pi*x - pi/2.))*cos(t*pi/8.)
    
    S = Smin + 0.5*(Smax - Smin)*(1 - cos(pi*x))*(1. - sin(t*pi/24.))
    
    return h, S


def space_manufactured_solution(sim):
    
    return manufactured_solution(sim = sim, time = 1.)


def time_manufactured_solution(sim):
    
    return manufactured_solution(sim = sim, time = sim.time)
    
    
def test__verify_spatial_convergence__second_order__via_mms(
        mesh_sizes = (2, 4, 8, 16),
        tolerance = 0.1):
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = space_manufactured_solution,
        meshes = [fe.UnitIntervalMesh(size) for size in mesh_sizes],
        sim_constructor_kwargs = {"quadrature_degree": q},
        parameters = {
            "pure_liquidus_temperature": T_m,
            "stefan_number": Ste,
            "lewis_number": Le,
            "heat_capacity_solid_to_liquid_ratio": c_sl,
            "thermal_conductivity_solid_to_liquid_ratio": k_sl},
        expected_order = 2,
        timestep_size = endtime,
        endtime = endtime,
        tolerance = tolerance,
        write_simulation_outputs = False)


def test__verify_temporal_convergence__first_order__via_mms(
        meshsize = 16,
        timestep_sizes = (endtime, endtime/2., endtime/4.),
        tolerance = 0.1):
    
    sapphire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = time_manufactured_solution,
        mesh = fe.UnitIntervalMesh(meshsize),
        sim_constructor_kwargs = {"quadrature_degree": q},
        parameters = {
            "pure_liquidus_temperature": T_m,
            "stefan_number": Ste,
            "lewis_number": Le,
            "heat_capacity_solid_to_liquid_ratio": c_sl,
            "thermal_conductivity_solid_to_liquid_ratio": k_sl},
        expected_order = 1,
        endtime = endtime,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance)

    
def test__verify_temporal_convergence__second_order__via_mms(
        meshsize = 16,
        timestep_sizes = (endtime, endtime/2., endtime/4.),
        tolerance = 0.1):
    
    sapphire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = time_manufactured_solution,
        mesh = fe.UnitIntervalMesh(meshsize),
        sim_constructor_kwargs = {
            "element_degree": 1,
            "time_stencil_size": 3,
            "quadrature_degree": q},
        parameters = {
            "pure_liquidus_temperature": T_m,
            "stefan_number": Ste,
            "lewis_number": Le,
            "heat_capacity_solid_to_liquid_ratio": c_sl,
            "thermal_conductivity_solid_to_liquid_ratio": k_sl,
            "quadrature_degree": q},
        expected_order = 2,
        endtime = endtime,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance)
    