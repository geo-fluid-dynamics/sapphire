import firedrake as fe
import sapphire.simulations.convection_coupled_alloy_phasechange


initial_porosity = 1.

basesim_module = sapphire.simulations.convection_coupled_alloy_phasechange

BaseSim = basesim_module.Simulation
    
def initial_values(sim):
    
    w_0 = fe.Function(sim.solution.function_space())
    
    p, u, h, S = w_0.split()
    
    p = p.assign(0.)
    
    ihat, jhat = sim.unit_vectors()
    
    u = u.assign(0.*ihat + 0.*jhat)
    
    h = h.assign(basesim_module.enthalpy(
        sim = sim,
        temperature = sim.max_temperature,
        porosity = initial_porosity))
    
    S = S.assign(sim.initial_solute_concentration)
    
    return w_0

    
def dirichlet_boundary_conditions(sim):

    W = sim.function_space
    
    h_h = basesim_module.enthalpy(
        sim = sim,
        temperature = sim.max_temperature,
        porosity = initial_porosity)
    
    h_c = basesim_module.enthalpy(
        sim = sim,
        temperature = sim.cold_wall_temperature,
        porosity = sim.cold_wall_porosity)
    
    return [
        fe.DirichletBC(W.sub(1), (0., 0.), 1),
        fe.DirichletBC(W.sub(1), (0., 0.), 2),
        fe.DirichletBC(W.sub(2), h_h, 1),
        fe.DirichletBC(W.sub(2), h_c, 2)]
        
        
class Simulation(BaseSim):

    def __init__(self, *args, 
            nx,
            ny,
            Lx,
            Ly,
            cold_wall_temperature,
            initial_solute_concentration,
            cold_wall_porosity,
            **kwargs):
        
        self.cold_wall_temperature = fe.Constant(cold_wall_temperature)
        
        self.initial_solute_concentration = fe.Constant(initial_solute_concentration)
        
        self.cold_wall_porosity = fe.Constant(cold_wall_porosity)
        
        super().__init__(
            *args,
            mesh = fe.PeriodicRectangleMesh(nx, ny, Lx, Ly, direction = "x"),
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        