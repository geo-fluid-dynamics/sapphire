""" Gallium melting benchmark 

Physical parameters are based on 

    @article{belhamadia2019adaptive,
        author = {Belhamadia, Youssef and Fortin, André and Briffard, Thomas},
        year = {2019},
        month = {06},
        pages = {1-19},
        title = {A two-dimensional adaptive remeshing method for solving melting and solidification problems with convection},
        volume = {76},
        journal = {Numerical Heat Transfer, Part A: Applications},
        doi = {10.1080/10407782.2019.1627837},
    }
"""
import firedrake as fe
import sapphire.simulations.convection_coupled_phasechange


reference_length = 6.35  # cm

reference_temperature = 301.3  # K

reference_temperature_range = 9.7  # K

reference_time = 292.90  # s


dimensionless_liquidus_temperature = 0.1525

prandtl_number = 0.0216

reynolds_number = 1./prandtl_number

rayleigh_number = 7.e5

stefan_number = 0.046

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
            horizontal_edges,
            vertical_edges,
            cutoff_length = 0.5,
            **kwargs):
        
        self.hotwall_temperature = fe.Constant(1.)
        
        self.coldwall_temperature = fe.Constant(0.)
        
        self.initial_temperature = fe.Constant(self.coldwall_temperature)
        
        super().__init__(
            *args,
            liquidus_temperature = dimensionless_liquidus_temperature,
            reynolds_number = reynolds_number,
            rayleigh_number = rayleigh_number,
            prandtl_number = prandtl_number,
            stefan_number = stefan_number,
            mesh = fe.RectangleMesh(
                horizontal_edges,
                vertical_edges,
                cutoff_length,
                1.),
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        