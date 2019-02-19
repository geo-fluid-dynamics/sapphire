""" An enthalpy model class for melting and solidification """
import firedrake as fe
import fempy.unsteady_model

    
class Model(fempy.unsteady_model.Model):
    
    def __init__(self, quadrature_degree, spatial_order, temporal_order):
        
        self.stefan_number = fe.Constant(1.)
        
        self.liquidus_temperature = fe.Constant(0.)
        
        self.smoothing = fe.Constant(1./32.)
        
        super().__init__(
            quadrature_degree = quadrature_degree,
            spatial_order = spatial_order,
            temporal_order = temporal_order)
        
    def init_element(self):
    
        self.element = fe.FiniteElement(
            "P", self.mesh.ufl_cell(), self.spatial_order - 1)
        
    def porosity(self, T):
        
        T_L = self.liquidus_temperature
        
        s = self.smoothing
        
        tanh = fe.tanh
        
        return 0.5*(1. + tanh((T - T_L)/s))
    
    def init_time_discrete_terms(self):
        
        super().init_time_discrete_terms()
        
        solutions = [self.solution]
        
        for iv in self.initial_values:
        
            solutions.append(iv)
            
        phil_t = fempy.time_discretization.bdf(
            [self.porosity(T_n) for T_n in solutions],
            order = self.temporal_order,
            timestep_size = self.timestep_size)
        
        self.time_discrete_terms = (self.time_discrete_terms, phil_t)
    
    def init_weak_form_residual(self):
        
        T = self.solution
        
        Ste = self.stefan_number
        
        T_t, phil_t = self.time_discrete_terms
        
        v = fe.TestFunction(self.function_space)
        
        dot, grad = fe.dot, fe.grad
        
        self.weak_form_residual = v*(T_t + 1./Ste*phil_t) +\
            dot(grad(v), grad(T))
        
    def init_solver(self, solver_parameters = {"ksp_type": "cg"}):
        
        self.solver = fe.NonlinearVariationalSolver(
            self.problem, solver_parameters = solver_parameters)
    