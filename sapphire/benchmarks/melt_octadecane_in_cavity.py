import firedrake as fe
import sapphire.simulations.convection_coupled_phasechange


def initial_values(sim):
    
    w = fe.Function(sim.function_space)
    
    p, u, T = w.split()
    
    p.assign(0.)
    
    ihat, jhat = sim.unit_vectors()
    
    u.assign(0.*ihat + 0.*jhat)
    
    T.assign(sim.initial_temperature)
    
    return w
    
    
def dirichlet_boundary_conditions(sim):

    W = sim.function_space
    
    return [
        fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
        fe.DirichletBC(W.sub(2), sim.hotwall_temperature, 1),
        fe.DirichletBC(W.sub(2), sim.initial_temperature, 2)]
        
        
class Simulation(sapphire.simulations.\
        convection_coupled_phasechange.Simulation):
    
    def __init__(self, *args, 
            meshsize,
            hotwall_temperature = 1.,
            initial_temperature = -0.01, 
            stefan_number = 0.045,
            rayleigh_number = 3.27e5,
            prandtl_number = 56.2,
            liquidus_temperature = 0.,
            **kwargs):
        
        self.hotwall_temperature = fe.Constant(hotwall_temperature)
        
        self.initial_temperature = fe.Constant(initial_temperature)
        
        grashof_number = rayleigh_number/prandtl_number
        
        super().__init__(
            *args,
            liquidus_temperature = liquidus_temperature,
            stefan_number = stefan_number,
            grashof_number = grashof_number,
            prandtl_number = prandtl_number,
            mesh = fe.UnitSquareMesh(meshsize, meshsize),
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        