""" An enthalpy model class for melting and solidification """
import firedrake as fe
import fempy.unsteady_model

    
class Model(fempy.unsteady_model.Model):
    
    def __init__(self):
        
        self.stefan_number = fe.Constant(1.)
        
        self.liquidus_temperature = fe.Constant(0.)
        
        self.latent_heat_smoothing = fe.Constant(1./32.)
        
        super().__init__()
        
    def init_element(self):
    
        self.element = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        
    def porosity(self, T):
        
        T_L = self.liquidus_temperature
        
        s = self.latent_heat_smoothing
        
        tanh = fe.tanh
        
        return 0.5*(1. + tanh((T - T_L)/s))
    
    def init_time_discrete_terms(self):
        """ Implicit Euler finite difference scheme """
        T = self.solution
        
        T_n = self.initial_values
        
        Delta_t = self.timestep_size
        
        T_t = (T - T_n)/Delta_t
        
        phil = self.porosity
        
        phil_t = (phil(T) - phil(T_n))/Delta_t
        
        self.time_discrete_terms = T_t, phil_t
    
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
    