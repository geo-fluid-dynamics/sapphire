import firedrake as fe
import sapphire.simulations.alloy_phasechange

    
def initial_values(sim):
    
    w_0 = fe.Function(sim.function_space)
    
    h_0, S_l_0 = w_0.split()
    
    initial_enthalpy = \
        sapphire.simulations.alloy_phasechange.enthalpy(
            sim = sim,
            temperature = sim.initial_temperature,
            porosity = sim.initial_porosity)
    
    h_0 = h_0.assign(initial_enthalpy)
    
    S_l_0 = S_l_0.assign(sim.initial_liquid_solute_concentration)
    
    return w_0
    
    
def dirichlet_boundary_conditions(sim):
    
    T_c = sim.cold_boundary_temperature
    
    phi_lc = sim.cold_boundary_porosity
    
    h_c = sapphire.simulations.alloy_phasechange.enthalpy(
        sim = sim,
        temperature = T_c,
        porosity = phi_lc)
    
    T_m = sim.pure_liquidus_temperature
    
    S_lc = 1. - T_c/T_m  # Mushy layer, T = T_L(S_l) = T_m*(1 - S_l)
    
    W = sim.function_space
    
    return [
        fe.DirichletBC(W.sub(0), h_c, 1),
        fe.DirichletBC(W.sub(1), S_lc, 1)]
    

class Simulation(sapphire.simulations.alloy_phasechange.Simulation):
    
    def __init__(self, *args, 
            initial_liquid_solute_concentration,
            cold_boundary_temperature,
            cold_boundary_porosity,
            mesh_cellcount, 
            cutoff_length,
            initial_porosity = 1.,
            **kwargs):
        
        self.initial_liquid_solute_concentration = fe.Constant(
            initial_liquid_solute_concentration)
        
        self.initial_porosity = fe.Constant(initial_porosity)
            
        self.cold_boundary_temperature = fe.Constant(cold_boundary_temperature)
        
        self.cold_boundary_porosity = fe.Constant(cold_boundary_porosity)
        
        super().__init__(
            *args,
            mesh = fe.IntervalMesh(mesh_cellcount, cutoff_length),
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        