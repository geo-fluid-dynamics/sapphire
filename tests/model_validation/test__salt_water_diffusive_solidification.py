import firedrake as fe 
import sapphire.simulations.alloy_phasechange
import sapphire.benchmarks.diffusive_solidification_of_alloy
from math import sqrt


BaseSim = sapphire.benchmarks.diffusive_solidification_of_alloy.Simulation 

class SimWithoutSolutionOutput(BaseSim):
    """ Redefine output to skip solution writing and plotting 
        (which otherwise slows down the test)
    """
    def write_outputs(self, write_headers, plotvars = None):
        
        pass
        
        
T_m__degC = 0.  # [deg C]
    
T_e__degC = -32. # [deg C]

T_i__degC = 15.  # [deg C]

def T__nondim(T__degC):
    
    return (T__degC - T_e__degC)/(T_i__degC - T_e__degC)
    

S_e__wtpercNaNO3 = 80.  # [% wt. NaNO3]

def S__nondim(S__wtpercNaNO3):

    return S__wtpercNaNO3/S_e__wtpercNaNO3
    

k_l = 0.54  # [W/(m K)]

c_l = 4200.  # [J/(kg K)]

rho = 1000.  # [kg/m^3]

h_m = 3.3e5  # [J/m^3]

c_sl = 1.

D_l = 1.e-9  # m^2/s


T_c__degC = -18.6  # [deg C]

S_l0__wtpercNaNO3 = 14.  # [% wt. NaNO3]


endtime = 0.05

nt = 200


alpha_l = k_l/(rho*c_l)

Le = alpha_l/D_l

Ste = c_l*(T_i__degC - T_e__degC)/h_m

T_m = T__nondim(T_m__degC)

T_c = T__nondim(T_c__degC)

S_l0 = S__nondim(S_l0__wtpercNaNO3)

Delta_t = endtime/nt

def test__compare_to_lebars2006():
    
    print("alpha_l = {} [m^2/s]".format(alpha_l))
    
    print("Ste = {}".format(Ste))
    
    print("Le = {}".format(Le))
    
    print("T_m = {}".format(T_m))
    
    print("S_l0 = {}".format(S_l0))
    
    print("T_c = {}".format(T_c))
    
    print("Delta_t = {}".format(Delta_t))
    
    sim = SimWithoutSolutionOutput(
        lewis_number = Le,
        stefan_number = Ste,
        initial_liquid_solute_concentration = S_l0,
        pure_liquidus_temperature = T_m,
        porosity_smoothing = 0.001,
        cold_boundary_temperature = T_c,
        cold_boundary_porosity = 0.001,
        quadrature_degree = 2,
        mesh_cellcount = 100,
        cutoff_length = 1.,
        timestep_size = Delta_t,
        output_directory_path = "salt_water_diffusive_solidification/")
    
    sim.solutions, sim.time = sim.run(endtime = endtime)
    
    h, S_l = sim.solution.split()
        
    phi_l = fe.interpolate(
        sapphire.simulations.alloy_phasechange.regularized_porosity(
            sim = sim,
            enthalpy = h,
            liquid_solute_concentration = S_l),
        sim.postprocessing_function_space)
        
    x_ml = sapphire.benchmarks.diffusive_solidification_of_alloy.\
        find_mush_liquid_interface_position(
            positions = sim.mesh.coordinates.vector().array(),
            porosities = phi_l.vector().array(),
            interface_porosity = 0.999)
    
    print("x_ml = {}".format(x_ml))
    
    t = sim.time.__float__()
    
    lambda_b = x_ml/(2.*sqrt(t/Le))
    
    print("lambda_b = {}".format(lambda_b))
    
    expected_lambda_b = 3.
    
    tolerance = 0.1
    
    assert(abs(lambda_b - expected_lambda_b) < tolerance)


if __name__ == "__main__":
    
    test__compare_to_lebars2006()
    