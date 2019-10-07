import firedrake as fe 
import sapphire.mms
from sapphire.simulations import alloy_phasechange as sim_module


T_m__degC = 0.

T_e__degC = -32.

T_i__degC = 15.

T_m = (T_m__degC - T_e__degC)/(T_i__degC - T_e__degC)

Ste = 0.6

Le = 80.

c_sl = 0.5

k_sl = 4.

phil_min = 0.01

def c(phi_l):

    return c_sl + (1. - c_sl)*phi_l


Tmin = 0.

hmin = c(phil_min)*(Tmin - T_m) + 1./Ste*phil_min

hmax = 1. - T_m + 1./Ste

m = -T_m

Smin = (hmin - phil_min/Ste)/((1. - c_sl)*m + c_sl*m/phil_min)

Smax = 1.

def manufactured_solution(sim, time):
    
    x = fe.SpatialCoordinate(sim.mesh)[0]
    
    t = time
    
    cos, sin, pi = fe.cos, fe.sin, fe.pi
    
    h = hmin + 0.5*(hmax - hmin)*(1 + sin(pi*x - pi/2.))*cos(t*pi/8.)
    
    S = Smin + 0.5*(Smax - Smin)*(1 - cos(pi*x))*(1. - sin(t*pi/24.))
    
    return h, S

    
spatial_endtime = 0.0001

def space_manufactured_solution(sim):
    
    return manufactured_solution(sim = sim, time = spatial_endtime)


def test__verify_spatial_convergence__second_order__via_mms(
        mesh_sizes = (2, 4, 8, 16),
        tolerance = 0.1):
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = space_manufactured_solution,
        meshes = [fe.UnitIntervalMesh(size) for size in mesh_sizes],
        sim_constructor_kwargs = {
            "quadrature_degree": 2,
            "pure_liquidus_temperature": T_m,
            "stefan_number": Ste,
            "lewis_number": Le,
            "heat_capacity_solid_to_liquid_ratio": c_sl,
            "thermal_conductivity_solid_to_liquid_ratio": k_sl},
        expected_order = 2,
        timestep_size = spatial_endtime,
        endtime = spatial_endtime,
        tolerance = tolerance,
        write_simulation_outputs = False)


temporal_endtime = 0.1

def time_manufactured_solution(sim):
    
    return manufactured_solution(sim = sim, time = sim.time)
    
    
def test__verify_temporal_convergence__first_order__via_mms(
        meshsize = 128,
        timestep_sizes = [temporal_endtime/nt for nt in (4, 8, 16)],
        tolerance = 0.1):
    
    sapphire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        manufactured_solution = time_manufactured_solution,
        mesh = fe.UnitIntervalMesh(meshsize),
        sim_constructor_kwargs = {
            "quadrature_degree": 2,
            "pure_liquidus_temperature": T_m,
            "stefan_number": Ste,
            "lewis_number": Le,
            "heat_capacity_solid_to_liquid_ratio": c_sl,
            "thermal_conductivity_solid_to_liquid_ratio": k_sl},
        expected_order = 1,
        endtime = temporal_endtime,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance,
        write_simulation_outputs = False)
