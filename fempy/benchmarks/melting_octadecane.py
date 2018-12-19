import firedrake as fe
import fempy.models.enthalpy_porosity


class Model(fempy.models.enthalpy_porosity.Model):
    
    def __init__(self, meshsize):
        
        self.meshsize = meshsize
        
        self.hot_wall_temperature = fe.Constant(1.)
    
        self.cold_wall_temperature = fe.Constant(-0.01)
        
        super().__init__()
        
        self.rayleigh_number.assign(3.27e5)
        
        self.prandtl_number.assign(56.2)
        
        self.stefan_number.assign(0.045)
        
        self.liquidus_temperature.assign(0.)

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

    def run_and_plot(self, endtime):
        
        output_prefix = self.output_prefix
        
        while self.time.__float__() < (endtime - self.time_tolerance):
        
            self.run(endtime = self.time.__float__() + self.timestep_size.__float__())
            
            self.output_prefix = output_prefix + "t" + str(self.time.__float__()) + "_"
            
            self.plot(save = True, show = False)
    
if __name__ == "__main__":
    
    model = Model(meshsize = 32)
    
    model.timestep_size.assign(10.)
    
    model.latent_heat_smoothing.assign(1./256.)
    
    model.run_and_plot(endtime = 70.)
    