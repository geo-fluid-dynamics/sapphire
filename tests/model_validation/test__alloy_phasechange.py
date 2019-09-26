import firedrake as fe 
import sapphire.benchmarks.diffusive_solidification_of_alloy


def test__validate__diffusive_solidification():
    
    endtime = 1.
    
    mesh_cellcount = 1000
    
    timestep_count = 100000
    
    quadrature_degree = 4
    
    
    T_m = 0.  # [deg C]
    
    T_e = -32.  # [deg C]
    
    T_i = 15.  # [deg C]
    
    def T(T__degC):
        
        return (T__degC - T_e)/(T_i - T_e)
    
    
    k = 0.5442  # [W/(m K)]
    
    rho = 1000.  # [kg/m^3]
    
    C_p = 4.186e6  # [J/(m^3 K)]
    
    c_p = C_p/rho
    
    alpha = k/(rho*c_p)
    
    D = 1.e-9  # [m^2/s]
    
    Le = alpha/D
    
    h_m = 3.3488e8  # [J/m^3]
    
    Ste = c_p*(T_i - T_e)/h_m
    
    S_e = 0.80
    
    def S(S__wtperc):
    
        return S__wtperc/S_e
    
    
    sim = sapphire.benchmarks.diffusive_solidification_of_alloy.Simulation(
        lewis_number = Le,
        stefan_number = Ste,
        farfield_concentration = S(0.14),
        pure_liquidus_temperature = T(0.),
        cold_boundary_temperature = T(-18.6),
        quadrature_degree = quadrature_degree,
        mesh_cellcount = mesh_cellcount,
        output_directory_path = "output/diffusive_solidification/" 
            + "nx{}_nt{}_q{}".format(
                mesh_cellcount, timestep_count, quadrature_degree))
    
    
    sim.timestep_size.assign(endtime/float(timestep_count))
    
    sim.solutions, sim.time, = sim.run(endtime = endtime)
