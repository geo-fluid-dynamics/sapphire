import firedrake as fe
import sapphire.simulations.convection_coupled_phasechange


def initial_values(sim):
    
    w = fe.Function(sim.function_space)
    
    p, u, T = w.split()
    
    p.assign(0.)
    
    ihat, jhat = sim.unit_vectors()
    
    u.assign(0.*ihat + 0.*jhat)
    
    T.assign(sim.cold_wall_temperature)
    
    return w
    
    
def dirichlet_boundary_conditions(sim):

    W = sim.function_space
    
    return [
        fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
        fe.DirichletBC(W.sub(2), sim.hot_wall_temperature, 1),
        fe.DirichletBC(W.sub(2), sim.cold_wall_temperature, 2)]
        
        
class Simulation(sapphire.simulations.\
        convection_coupled_phasechange.Simulation):
    
    def __init__(self, *args, meshsize, **kwargs):
        
        self.hot_wall_temperature = fe.Constant(1.)
    
        self.cold_wall_temperature = fe.Constant(-0.01)
        
        self.topwall_heatflux = fe.Constant(0.)
        
        super().__init__(
            *args,
            mesh = fe.UnitSquareMesh(meshsize, meshsize),
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        
        q = self.topwall_heatflux

        _, _, psi_T = fe.TestFunctions(self.function_space)

        ds = fe.ds(domain = self.mesh, subdomain_id = 4)

        self.variational_form_residual += psi_T*q*ds
        
        Ra = 3.27e5
        
        Pr = 56.2
        
        self.grashof_number = self.grashof_number.assign(Ra/Pr)
        
        self.prandtl_number = self.prandtl_number.assign(Pr)
        
        self.stefan_number = self.stefan_number.assign(0.045)
        
        self.liquidus_temperature = self.liquidus_temperature.assign(0.)
        
    def run(self, *args,
            endtime,
            topwall_heatflux_poststart = -0.02,
            topwall_heatflux_starttime = 40.,
            **kwargs):
    
        final_endtime = endtime
        
        original_topwall_heatflux = self.topwall_heatflux.__float__()
        
        if final_endtime < topwall_heatflux_starttime:
        
            self.solutions, self.time = super().run(*args,
                endtime = final_endtime,
                **kwargs)
            
            return self.solutions, self.time
        
        self.solutions, self.time = super().run(*args,
            endtime = topwall_heatflux_starttime,
            write_initial_outputs = True,
            **kwargs)
        
        self.topwall_heatflux = self.topwall_heatflux.assign(
            topwall_heatflux_poststart)
            
        self.solutions, self.time = super().run(*args,
            endtime = final_endtime,
            write_initial_outputs = False,
            **kwargs)
        
        return self.solutions, self.time
        