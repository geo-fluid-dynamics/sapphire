""" An enthalpy formulated phase-change model class """
import firedrake as fe
import fempy.unsteady_model

    
class Model(fempy.unsteady_model.UnsteadyModel):
    
    def __init__(self):
        
        self.stefan_number = fe.Constant(1.)
        
        self.normalized_liquidus_temperature = fe.Constant(0.)
        
        self.phase_interface_smoothing = fe.Constant(1./32.)
        
        super().__init__()
        
    def init_element(self):
    
        self.element = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        
    def semi_phasefield(self, theta):
        
        theta_L = self.normalized_liquidus_temperature
        
        r = self.phase_interface_smoothing
        
        tanh = fe.tanh
        
        return 0.5*(1. + tanh((theta_L - theta)/r))
    
    def init_time_discrete_terms(self):
        """ Implicit Euler finite difference scheme """
        theta = self.solution
        
        thetan = self.initial_values[0]
        
        Delta_t = self.timestep_size
        
        theta_t = (theta - thetan)/Delta_t
        
        phi = self.semi_phasefield
        
        phi_t = (phi(theta) - phi(thetan))/Delta_t
        
        self.time_discrete_terms = theta_t, phi_t
    
    def init_weak_form_residual(self):
        
        theta = self.solution
        
        Ste = self.stefan_number
        
        self.init_time_discrete_terms()
        
        theta_t, phi_t = self.time_discrete_terms
        
        v = fe.TestFunction(self.function_space)
        
        dot, grad = fe.dot, fe.grad
        
        self.weak_form_residual = v*(theta_t - 1./Ste*phi_t) + dot(grad(v), grad(theta))
        
    def init_solver(self, solver_parameters = {"ksp_type": "cg"}):
        
        self.solver = fe.NonlinearVariationalSolver(
            self.problem, solver_parameters = solver_parameters)
    