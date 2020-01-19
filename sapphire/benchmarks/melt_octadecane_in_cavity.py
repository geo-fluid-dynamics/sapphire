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
            topwall_heatflux_prestart = 0.,
            topwall_heatflux_poststart = 0.,
            topwall_heatflux_starttime = 40.,
            stefan_number = 0.045,
            rayleigh_number = 3.27e5,
            prandtl_number = 56.2,
            liquidus_temperature = 0.,
            **kwargs):
        
        self.hotwall_temperature = fe.Constant(hotwall_temperature)
        
        self.initial_temperature = fe.Constant(initial_temperature)
        
        self.topwall_heatflux = fe.Constant(topwall_heatflux_prestart)
        
        self.topwall_heatflux_poststart = topwall_heatflux_poststart
        
        self.topwall_heatflux_starttime = topwall_heatflux_starttime
        
        Ra = rayleigh_number
        
        Pr = prandtl_number
        
        Gr = Ra/Pr
        
        super().__init__(
            *args,
            liquidus_temperature = liquidus_temperature,
            stefan_number = stefan_number,
            grashof_number = Gr,
            prandtl_number = prandtl_number,
            mesh = fe.UnitSquareMesh(meshsize, meshsize),
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        
        q = self.topwall_heatflux

        _, _, psi_T = fe.TestFunctions(self.function_space)
        
        topwall_id = 4
        
        ds = fe.ds(domain = self.mesh, subdomain_id = topwall_id)

        self.variational_form_residual += psi_T*q*ds
        
    def run(self, *args, endtime, **kwargs):
    
        final_endtime = endtime
        
        original_topwall_heatflux = self.topwall_heatflux.__float__()
        
        if final_endtime < self.topwall_heatflux_starttime:
        
            self.solutions, self.time = super().run(*args,
                endtime = final_endtime,
                **kwargs)
            
            return self.solutions, self.time
        
        self.solutions, self.time = super().run(*args,
            endtime = self.topwall_heatflux_starttime,
            write_initial_outputs = True,
            **kwargs)
        
        self.topwall_heatflux = self.topwall_heatflux.assign(
            self.topwall_heatflux_poststart)
            
        self.solutions, self.time = super().run(*args,
            endtime = final_endtime,
            write_initial_outputs = False,
            **kwargs)
        
        return self.solutions, self.time
        