import firedrake as fe
import sapphire.simulations.alloy_phasechange

    
def farfield_enthalpy(sim):
    
    return sapphire.simulations.alloy_phasechange.enthalpy(
        sim = sim,
        temperature = sim.farfield_temperature,
        porosity = 1.)


def initial_values(sim):
    
    w0 = fe.Function(sim.function_space)
    
    h0, S0 = w0.split()
    
    h_h = farfield_enthalpy(sim = sim)
    
    h0 = h0.assign(h_h)
    
    S0 = S0.assign(sim.farfield_concentration)
    
    return w0
    
    
def dirichlet_boundary_conditions(sim):
    
    phi_lc = sim.cold_boundary_porosity
    
    h_c = sapphire.simulations.alloy_phasechange.enthalpy(
        sim = sim,
        temperature = sim.cold_boundary_temperature,
        porosity = phi_lc)
    
    Ste = sim.stefan_number
    
    c_sl = sim.heat_capacity_solid_to_liquid_ratio
    
    T_m = sim.pure_liquidus_temperature
    
    m = -T_m
    
    S_c = (h_c - phi_lc/Ste)/((1. - c_sl)*m + c_sl*m/phi_lc)
    
    h_h = farfield_enthalpy(sim = sim)
    
    S_h = sim.farfield_concentration
    
    W = sim.function_space
    
    return [
        fe.DirichletBC(W.sub(0), h_c, 1),
        fe.DirichletBC(W.sub(1), S_c, 1),
        fe.DirichletBC(W.sub(0), h_h, 2),
        fe.DirichletBC(W.sub(1), S_h, 2)]
    

class Simulation(sapphire.simulations.alloy_phasechange.Simulation):
    
    def __init__(self, *args, 
            farfield_concentration,
            pure_liquidus_temperature,
            cold_boundary_temperature,
            cold_boundary_porosity = 0.01,
            mesh_cellcount = 100, 
            cutoff_length = 1., 
            **kwargs):
        
        self.farfield_concentration = fe.Constant(farfield_concentration)
        
        self.cold_boundary_temperature = fe.Constant(cold_boundary_temperature)
        
        self.cold_boundary_porosity = fe.Constant(cold_boundary_porosity)
        
        super().__init__(
            *args,
            mesh = fe.IntervalMesh(mesh_cellcount, cutoff_length),
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        