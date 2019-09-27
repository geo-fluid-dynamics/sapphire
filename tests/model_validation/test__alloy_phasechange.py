import firedrake as fe 
import sapphire.benchmarks.diffusive_solidification_of_alloy


class DebugSim(sapphire.benchmarks.diffusive_solidification_of_alloy.Simulation):
    
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
    """    
    
def test__validate__diffusive_solidification():
    
    endtime = t_f = 1.
    
    cutoff_length = xmax = 1.
    
    mesh_cellcount = nx = 40
    
    timestep_count = nt = 200
    
    quadrature_degree = q = 2
    
    
    T_m = 0.  # [deg C]
    
    T_e = -21.1  # [deg C]
    
    T_h = T_m # [deg C]
    
    def T(T__degC):
        
        return (T__degC - T_e)/(T_h - T_e)
    
    
    #k = 0.5442  # [W/(m K)]
    
    rho = 1000.  # [kg/m^3]
    
    C_p = 4.186e6  # [J/(m^3 K)]
    
    c_p = C_p/rho # [J/(kg K)]
    
    #alpha = k/(rho*c_p)
    
    #D = 1.e-9  # [m^2/s]
    
    #Le = alpha/D
    Le = 80.
    
    h_m = 3.3488e8  # [J/m^3]
    
    Ste = C_p*(T_h - T_e)/h_m
    
    S_e = 0.27
    
    def S(S__wtperc):
    
        return S__wtperc/S_e
    
    
    S_h = 0.035
    
    phi_lc = 0.
    
    T_c = T_e
    
    outdir_path = "output/diffusive_solidification/"\
    + "Le{}_Ste{}_Te{}_Se{}_Tc{}_Th{}_Sh{}"\
    + "__tf{}_philc{}_xmax{}_nx{}_nt{}_q{}"
    
    outdir_path = outdir_path.format(
        Le, Ste, T_e, S_e, T_c, T_h, S_h,
        t_f, phi_lc, xmax, nx, nt, q)
    
    sim = DebugSim(
        lewis_number = Le,
        stefan_number = Ste,
        farfield_concentration = S(S_h),
        pure_liquidus_temperature = T(T_m),
        cold_boundary_temperature = T(T_c),
        cold_boundary_porosity = phi_lc,
        quadrature_degree = q,
        mesh_cellcount = nx,
        cutoff_length = xmax,
        output_directory_path = outdir_path)
    
    sim.timestep_size.assign(endtime/float(timestep_count))
    
    sim.solutions, sim.time, = sim.run(endtime = endtime)
