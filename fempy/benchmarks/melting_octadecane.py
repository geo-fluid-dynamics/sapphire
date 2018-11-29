import firedrake as fe
import fempy.models.convection_coupled_phasechange


class Model(fempy.models.convection_coupled_phasechange.Model):
    
    def __init__(self, meshsize):
        
        self.meshsize = meshsize
        
        self.hot_wall_temperature = fe.Constant(1.)
    
        self.cold_wall_temperature = fe.Constant(-0.01)
        
        super().__init__()
        
        model.rayleigh_number.assign(3.27e5)
        
        model.prandtl_number.assign(56.2)
        
        model.stefan_number.assign(0.045)
        
        model.liquidus_temperature.assign(0.)

    def init_mesh(self):
    
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def assign_initial_values(self):
        
        initial_values = fe.interpolate(
            fe.Expression(
                (0., 0., 0., self.cold_wall_temperature.__float__()),
                element = self.element),
            self.function_space)
        
        self.solution.assign(initial_values)
        
        self.initial_values[0].assign(initial_values)
        
    def init_solution(self):
        
        super().init_solution()
        
        self.assign_initial_values()
        
    def init_dirichlet_boundary_conditions(self):
    
        W = self.function_space
        
        self.dirichlet_boundary_conditions = [
            fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
            fe.DirichletBC(W.sub(2), self.hot_wall_temperature, 1),
            fe.DirichletBC(W.sub(2), self.cold_wall_temperature, 2)]
    