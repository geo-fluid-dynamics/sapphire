import firedrake as fe
import fempy.models.convection_coupled_phasechange


class Model(fempy.models.convection_coupled_phasechange.Model):
    
    def __init__(self, meshsize):
        
        self.meshsize = meshsize
        
        self.hot_wall_temperature = fe.Constant(1.)
    
        self.cold_wall_temperature = fe.Constant(-0.01)
        
        super().__init__()
        
        self.timestep_size.assign(10.)
        
        self.rayleigh_number.assign(3.27e5)
        
        self.prandtl_number.assign(56.2)
        
        self.stefan_number.assign(0.045)
        
        self.liquidus_temperature.assign(0.01)
        
        self.phase_interface_smoothing.assign(1./32.)
        
        self.smoothing_sequence = (1./2., 1./4., 1./8., 1./16., 1./32.)
        
    def init_mesh(self):
    
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def assign_initial_values(self):
        
        initial_values = fe.interpolate(
            fe.Expression(
                (0., 0., 0., 0.),
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
            
    def run_timestep(self):
    
        assert(self.phase_interface_smoothing.__float__() == \
            self.smoothing_sequence[-1])
    
        self.initial_values[0].assign(self.solution)
        
        self.time.assign(self.time + self.timestep_size)
        
        for s in self.smoothing_sequence:
        
            print("Solving with s = " + str(s))
            
            self.phase_interface_smoothing.assign(s)
            
            self.solver.solve()
    