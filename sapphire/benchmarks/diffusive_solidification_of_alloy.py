import firedrake as fe
import sapphire.simulations.alloy_phasechange

    
def initial_values(sim):
    
    w0 = fe.Function(sim.function_space)
    
    h0, S0 = w0.split()
    
    assumed_initial_porosity = 1.
    
    constant_initial_enthalpy = \
        sapphire.simulations.alloy_phasechange.enthalpy(
            sim = sim,
            temperature = sim.initial_temperature,
            porosity = assumed_initial_porosity)
    
    h0 = h0.assign(constant_initial_enthalpy)
    
    S0 = S0.assign(sim.initial_concentration)
    
    constant_initial_porosity = \
        sapphire.simulations.alloy_phasechange.liquid_volume_fraction(
            sim = sim,
            enthalpy = constant_initial_enthalpy,
            solute_concentration = sim.initial_concentration)
    
    epsilon = 1.e-4
    
    phi_l0 = constant_initial_porosity.__float__()
    
    if abs(phi_l0 - assumed_initial_porosity) >= epsilon:
    
        raise ValueError(
            "For this test, it is assumed that the initial porosity is equal to {} +/- {}.".format(
                assumed_initial_porosity, epsilon)
            +"\nWhen setting initial values, the initial porosity was computed to be {}.".format(
                phi_l0))
    
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
    
    W = sim.function_space
    
    return [
        fe.DirichletBC(W.sub(0), h_c, 1),
        fe.DirichletBC(W.sub(1), S_c, 1)]
    

class Simulation(sapphire.simulations.alloy_phasechange.Simulation):
    
    def __init__(self, *args, 
            initial_concentration,
            cold_boundary_temperature,
            cold_boundary_porosity,
            mesh_cellcount, 
            cutoff_length, 
            **kwargs):
        
        self.initial_concentration = fe.Constant(initial_concentration)
        
        self.cold_boundary_temperature = fe.Constant(cold_boundary_temperature)
        
        self.cold_boundary_porosity = fe.Constant(cold_boundary_porosity)
        
        super().__init__(
            *args,
            mesh = fe.IntervalMesh(mesh_cellcount, cutoff_length),
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        