import firedrake as fe
import sapphire.simulations.convection_coupled_alloy_phasechange


basesim_module = sapphire.simulations.convection_coupled_alloy_phasechange

BaseSim = basesim_module.Simulation

initial_porosity = 1.

def initial_values(sim):
    
    w = fe.Function(sim.function_space)
    
    p, u, h, S_l = w.split()
    
    p = p.assign(0.)
    
    ihat, jhat = sim.unit_vectors()
    
    u = u.assign(0.*ihat + 0.*jhat)
    
    T_0 = sim.max_temperature
    
    h_0 = basesim_module.enthalpy(
        sim = sim,
        temperature = T_0,
        porosity = initial_porosity)
        
    h = h.assign(h_0)
    
    S_l = S_l.assign(sim.initial_solute_concentration)
    
    return w

    
def dirichlet_boundary_conditions(sim):

    W = sim.function_space
    
    T_c = sim.cold_wall_temperature
    
    h_0 = basesim_module.enthalpy(
        sim = sim,
        temperature = sim.max_temperature,
        porosity = initial_porosity)
    
    h_c = basesim_module.enthalpy(
        sim = sim,
        temperature = T_c,
        porosity = sim.cold_wall_porosity)
    
    T_m = sim.pure_liquidus_temperature
    
    S_lc = 1. - T_c/T_m  # Mushy layer, T = T_L(S_l) = T_m*(1 - S_l)
    
    leftwall, rightwall = 1, 2
    
    bottomwall, topwall = 3, 4
    
    return [
        fe.DirichletBC(W.sub(1), (0., 0.), topwall),
        fe.DirichletBC(W.sub(1), (0., 0.), leftwall),
        fe.DirichletBC(W.sub(1), (0., 0.), rightwall),
        fe.DirichletBC(W.sub(2), h_0, bottomwall),
        fe.DirichletBC(W.sub(2), h_c, topwall),
        fe.DirichletBC(W.sub(3), S_lc, topwall)]
        
        
class Simulation(BaseSim):

    def __init__(self, *args, 
            nx,
            ny,
            Lx,
            Ly,
            initial_solute_concentration,
            cold_wall_temperature,
            cold_wall_porosity,
            **kwargs):
        
        self.initial_solute_concentration = fe.Constant(initial_solute_concentration)
        
        self.cold_wall_temperature = fe.Constant(cold_wall_temperature)
        
        self.cold_wall_porosity = fe.Constant(cold_wall_porosity)
        
        super().__init__(
            *args,
            mesh = fe.RectangleMesh(nx, ny, Lx, Ly),
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        