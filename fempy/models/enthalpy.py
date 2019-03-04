""" An enthalpy model class for melting and solidification """
import firedrake as fe
import fempy.unsteady_model

    
class Model(fempy.unsteady_model.UnsteadyModel):
    
    def __init__(self, *args, mesh, element_degree, **kwargs):
        
        self.stefan_number = fe.Constant(1.)
        
        self.liquidus_temperature = fe.Constant(0.)
        
        self.smoothing = fe.Constant(1./32.)
        
        element = fe.FiniteElement("P", mesh.ufl_cell(), element_degree)
        
        super().__init__(*args, mesh, element, **kwargs)
        
    def porosity(self, T):
        
        T_L = self.liquidus_temperature
        
        s = self.smoothing
        
        tanh = fe.tanh
        
        return 0.5*(1. + tanh((T - T_L)/s))
    
    def init_time_discrete_terms(self):
        
        super().init_time_discrete_terms()
        
        phil_t = fempy.time_discretization.bdf(
            [self.porosity(T_n) for T_n in self.solutions],
            order = self.time_stencil_size - 1,
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
    
    def strong_form_residual(self, solution):
        
        T = solution
        
        t = self.time
        
        Ste = self.stefan_number
        
        phil = self.porosity
        
        diff, div, grad = fe.diff, fe.div, fe.grad
        
        return diff(T, t) - div(grad(T)) + 1./Ste*diff(phil(T), t)
    
    def init_solver(self, solver_parameters = {"ksp_type": "cg"}):
        
        self.solver = fe.NonlinearVariationalSolver(
            self.problem, solver_parameters = solver_parameters)
    