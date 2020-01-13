import firedrake as fe 
import sapphire.benchmarks.diffusive_solidification_of_alloy
from math import sqrt


class SimWithoutPlots(
        sapphire.benchmarks.diffusive_solidification_of_alloy.Simulation):
    """ Redefine output to skip plotting (which otherwise slows down the test)
    
    Solutions and post-processed functions are still written to VTK.
    """
    def write_outputs(self, write_headers, plotvars = None):
        
        if self.solution_file is None:
            
            solution_filepath = self.output_directory_path.joinpath(
                "solution").with_suffix(".pvd")
            
            self.solution_file = fe.File(str(solution_filepath))
        
        self = self.postprocess()
        
        sapphire.output.report(sim = self, write_header = write_headers)
        
        sapphire.output.write_solution(sim = self, file = self.solution_file)
        
        #sapphire.output.plot(sim = self, plotvars = plotvars)
    

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
    
    sim = SimWithoutPlots(
        lewis_number = Le,
        stefan_number = Ste,
        initial_liquid_solute_concentration = S_l0,
        pure_liquidus_temperature = T_m,
        porosity_smoothing = 0.001,
        cold_boundary_temperature = T_c,
        cold_boundary_porosity = 0.001,
        quadrature_degree = 2,
        mesh_cellcount = 1000,
        cutoff_length = 1.,
        timestep_size = Delta_t,
        snes_linesearch_damping = 1.,
        snes_max_iterations = 1000,
        snes_absolute_tolerance = 1.e-9,
        snes_step_tolerance = 1.e-9,
        snes_linesearch_maxstep = 1.,
        output_directory_path = "salt_water_diffusive_solidification/")
    
    sim.solutions, sim.time, = sim.run(endtime = 0.1)
    
    t = sim.time.__float__()
    
    x_ml = sim.mush_liquid_interface_position
    
    print("x_ml = {}".format(x_ml))
    
    lambda_b = x_ml/(2.*sqrt(t/Le))
    
    print("lambda_b = {}".format(lambda_b))
    
    expected_lambda_b = 3.
    
    tolerance = 0.1
    
    assert(abs(lambda_b -expected_lambda_b) < tolerance)


if __name__ == "__main__":
    
    test__compare_to_lebars2006()
    