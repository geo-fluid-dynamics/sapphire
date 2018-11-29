""" A convection-coupled phase-change model class """
import firedrake as fe
import fempy.unsteady_model

    
class Model(fempy.unsteady_model.Model):
    
    def __init__(self):
    
        self.solid_dynamic_viscosity = fe.Constant(1.e8)
        
        self.rayleigh_number = fe.Constant(1.)
        
        self.prandtl_number = fe.Constant(1.)
        
        self.stefan_number = fe.Constant(1.)
        
        self.pressure_penalty_factor = fe.Constant(1.e-7)
        
        self.liquidus_temperature = fe.Constant(0.)
        
        self.phase_interface_smoothing = fe.Constant(1./32.)
        
        self.smoothing_sequence = (1./2., 1./4., 1./8., 1./16., 1./32.)
        
        super().__init__()
        
    def init_element(self):
    
        self.element = fe.MixedElement(
            fe.FiniteElement("P", self.mesh.ufl_cell(), 1),
            fe.VectorElement("P", self.mesh.ufl_cell(), 2),
            fe.FiniteElement("P", self.mesh.ufl_cell(), 1))
    
    def semi_phasefield(self, T):
        """ Regularization from \cite{zimmerman2018monolithic} """
        T_L = self.liquidus_temperature
        
        s = self.phase_interface_smoothing
        
        tanh = fe.tanh
        
        return 0.5*(1. + tanh((T_L - T)/s))
        
    def buoyancy(self, T):
        """ Boussinesq buoyancy """
        Ra = self.rayleigh_number
        
        Pr = self.prandtl_number
        
        ihat, jhat = self.unit_vectors()
        
        ghat = fe.Constant(-jhat)
        
        return Ra/Pr*T*ghat
        
    def time_discrete_terms(self):
        """ Implicit Euler finite difference scheme """
        p, u, T = fe.split(self.solution)
        
        p_n, u_n, T_n = fe.split(self.initial_values[0])
        
        Delta_t = self.timestep_size
        
        u_t = (u - u_n)/Delta_t
        
        T_t = (T - T_n)/Delta_t
        
        phi = self.semi_phasefield
        
        phi_t = (phi(T) - phi(T_n))/Delta_t
        
        return u_t, T_t, phi_t
        
    def init_weak_form_residual(self):
        """ Weak form from \cite{zimmerman2018monolithic} """
        mu_S = self.solid_dynamic_viscosity
        
        mu_L = fe.Constant(1.)
        
        Ra = self.rayleigh_number
        
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        gamma = self.pressure_penalty_factor
        
        inner, dot, grad, div, sym = \
            fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
        p, u, T = fe.split(self.solution)
        
        u_t, T_t, phi_t = self.time_discrete_terms()
        
        b = self.buoyancy(T)
        
        phi = self.semi_phasefield(T)
        
        mu = mu_L + (mu_S - mu_L)*phi
        
        psi_p, psi_u, psi_T = fe.TestFunctions(self.solution.function_space())
        
        mass = psi_p*div(u)
        
        momentum = dot(psi_u, u_t + grad(u)*u + b) \
            - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), mu*sym(grad(u)))
        
        enthalpy = psi_T*(T_t - 1./Ste*phi_t) \
            + dot(grad(psi_T), 1./Pr*grad(T) - T*u)
        
        stabilization = gamma*psi_p*p
        
        self.weak_form_residual = mass + momentum + enthalpy + stabilization

    def run_timestep(self):
    
        assert(self.phase_interface_smoothing.__float__() == \
            self.smoothing_sequence[-1])
    
        self.initial_values[0].assign(self.solution)
        
        self.time.assign(self.time + self.timestep_size)
        
        for s in self.smoothing_sequence:
        
            print("Solving with s = " + str(s))
            
            self.phase_interface_smoothing.assign(s)
            
            self.solver.solve()
    