import firedrake as fe
import fempy.models.convection_coupled_phasechange


class Model(fempy.models.convection_coupled_phasechange.Model):
    
    def __init__(self, meshsize):
        
        self.meshsize = meshsize
        
        self.hot_wall_temperature = fe.Constant(1.)
    
        self.cold_wall_temperature = fe.Constant(-0.01)
        
        super().__init__()
        
        self.rayleigh_number.assign(3.27e5)
        
        self.prandtl_number.assign(56.2)
        
        self.stefan_number.assign(0.045)
        
        self.liquidus_temperature.assign(0.)
        
        self.phase_interface_smoothing.assign(1./256.)
        
        self.smoothing_sequence = (1./2., 1./4., 1./8., 1./16., 1./32.,
            1./64., 1./128., 1./256.)

    def init_mesh(self):
    
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def init_initial_values(self):
        
        self.initial_values = fe.interpolate(
            fe.Expression(
                (0., 0., 0., self.cold_wall_temperature.__float__()),
                element = self.element),
            self.function_space)
        
    def init_dirichlet_boundary_conditions(self):
    
        W = self.function_space
        
        self.dirichlet_boundary_conditions = [
            fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
            fe.DirichletBC(W.sub(2), self.hot_wall_temperature, 1),
            fe.DirichletBC(W.sub(2), self.cold_wall_temperature, 2)]
    
    
def run(meshsize, 
        timestep_size, 
        phase_interface_smoothing, 
        smoothing_sequence,
        endtime,
        plottimes):
    
    model = Model(meshsize = meshsize)
    
    model.assign_parameters({
        "timestep_size": timestep_size, 
        "phase_interface_smoothing": phase_interface_smoothing})
    
    model.smoothing_sequence = smoothing_sequence
    
    for plottime in plottimes:
        
        model.run(endtime = plottime)
    
        model.plot(prefix = "t" + str(model.time.__float__()) + "_")
        
    if plottime < (endtime - model.time_tolerance):
    
        model.run(endtime = endtime)
        
        model.plot(prefix = "t" + str(model.time.__float__()) + "_")
    
    return model
    
if __name__ == "__main__":
    
    run(
        meshsize = 32,
        timestep_size = 10.,
        phase_interface_smoothing = 1./256., 
        smoothing_sequence = (1./2., 1./4., 1./8., 1./16., 1./32.,
            1./64., 1./128., 1./256.),
        endtime = 70.,
        plottimes = (30., 50., 70.))
    