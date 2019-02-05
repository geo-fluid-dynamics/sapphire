import firedrake as fe
import fempy.models.enthalpy_porosity


class Model(fempy.models.enthalpy_porosity.Model):
    
    def __init__(self, meshsize):
        
        self.meshsize = meshsize
        
        self.hot_wall_temperature = fe.Constant(1.)
    
        self.cold_wall_temperature = fe.Constant(-0.01)
        
        super().__init__()
        
        Ra = 3.27e5
        
        Pr = 56.2
        
        self.grashof_number.assign(Ra/Pr)
        
        self.prandtl_number.assign(Pr)
        
        self.stefan_number.assign(0.045)
        
        self.liquidus_temperature.assign(0.)

        self.update_initial_values()
        
    def init_mesh(self):
    
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def init_dirichlet_boundary_conditions(self):
    
        W = self.function_space
        
        self.dirichlet_boundary_conditions = [
            fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
            fe.DirichletBC(W.sub(2), self.hot_wall_temperature, 1),
            fe.DirichletBC(W.sub(2), self.cold_wall_temperature, 2)]
            
    def update_initial_values(self):
        
        initial_values = fe.interpolate(
            fe.Expression(
                (0., 0., 0., self.cold_wall_temperature.__float__()),
                element = self.element),
            self.function_space)
            
        self.initial_values.assign(initial_values)

class ModelWithBDF2(fempy.models.enthalpy_porosity.ModelWithBDF2):
    
    def __init__(self, meshsize):
        
        self.meshsize = meshsize
        
        self.hot_wall_temperature = fe.Constant(1.)
    
        self.cold_wall_temperature = fe.Constant(-0.01)
        
        super().__init__()
        
        Ra = 3.27e5
        
        Pr = 56.2
        
        self.grashof_number.assign(Ra/Pr)
        
        self.prandtl_number.assign(Pr)
        
        self.stefan_number.assign(0.045)
        
        self.liquidus_temperature.assign(0.)

        self.update_initial_values()
        
    def init_mesh(self):
    
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def init_dirichlet_boundary_conditions(self):
    
        W = self.function_space
        
        self.dirichlet_boundary_conditions = [
            fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
            fe.DirichletBC(W.sub(2), self.hot_wall_temperature, 1),
            fe.DirichletBC(W.sub(2), self.cold_wall_temperature, 2)]
            
    def update_initial_values(self):
        
        initial_values = fe.interpolate(
            fe.Expression(
                (0., 0., 0., self.cold_wall_temperature.__float__()),
                element = self.element),
            self.function_space)
            
        if ((type(self.initial_values) == type((0,))) 
                or (type(self.initial_values) == type([0,]))):
                
            self.initial_values[0].assign(initial_values)
            
        else:
        
            self.initial_values.assign(initial_values)
            
        self.solution.assign(initial_values)
        
class ModelWithDarcyResistance(
        fempy.models.enthalpy_porosity.ModelWithDarcyResistance):
    
    def __init__(self, meshsize):
        
        self.meshsize = meshsize
        
        self.hot_wall_temperature = fe.Constant(1.)
    
        self.cold_wall_temperature = fe.Constant(-0.01)
        
        super().__init__()
        
        Ra = 3.27e5
        
        Pr = 56.2
        
        self.grashof_number.assign(Ra/Pr)
        
        self.prandtl_number.assign(Pr)
        
        self.stefan_number.assign(0.045)
        
        self.liquidus_temperature.assign(0.)

    def init_initial_values(self):
    
        super().init_initial_values()
        
        self.update_initial_values()
        
    def init_mesh(self):
    
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def init_dirichlet_boundary_conditions(self):
    
        W = self.function_space
        
        self.dirichlet_boundary_conditions = [
            fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
            fe.DirichletBC(W.sub(2), self.hot_wall_temperature, 1),
            fe.DirichletBC(W.sub(2), self.cold_wall_temperature, 2)]
            
    def update_initial_values(self):
        
        initial_values = fe.interpolate(
            fe.Expression(
                (0., 0., 0., self.cold_wall_temperature.__float__()),
                element = self.element),
            self.function_space)
            
        self.initial_values.assign(initial_values)
        
        
if __name__ == "__main__":
    
    model = Model(meshsize = 32)
    
    model.timestep_size.assign(10.)
    
    model.latent_heat_smoothing.assign(1./256.)
    
    model.run(endtime = 70., plot = True)
    