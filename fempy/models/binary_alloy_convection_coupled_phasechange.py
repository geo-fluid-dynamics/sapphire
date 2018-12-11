""" A binary alloy convection-coupled phase-change model class """
import firedrake as fe
import fempy.models.convection_coupled_phasechange
import matplotlib.pyplot as plt

    
class Model(fempy.models.convection_coupled_phasechange.Model):
    
    def __init__(self):
    
        self.temperature_rayleigh_number = fe.Constant(1.)
        
        self.concentration_rayleigh_number = fe.Constant(1.)
        
        self.schmidt_number = fe.Constant(1.)
        
        self.pure_liquidus_temperature = fe.Constant(0.)
        
        self.liquidus_slope = fe.Constant(-1.)
        
        super().__init__()
        
        delattr(self, "liquidus_temperature")
        
        delattr(self, "rayleigh_number")
        
    def init_element(self):
    
        P1 = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        
        P2 = fe.VectorElement("P", self.mesh.ufl_cell(), 2)
        
        self.element = fe.MixedElement(P1, P2, P1, P1)
    
    def liquidus_temperature(self, C):
    
        T_m = self.pure_liquidus_temperature
        
        m_L = self.liquidus_slope
        
        return T_m + m_L*C
    
    def semi_phasefield(self, T, C):
        """ Regularization from @cite{zimmerman2018monolithic} """
        T_L = self.liquidus_temperature
        
        s = self.phase_interface_smoothing
        
        tanh = fe.tanh
        
        return 0.5*(1. + tanh((T_L(C) - T)/s))
        
    def buoyancy(self, T, C):
        """ Boussinesq buoyancy """
        Ra_T = self.temperature_rayleigh_number
        
        Ra_C = self.concentration_rayleigh_number
        
        Pr = self.prandtl_number
        
        ihat, jhat = self.unit_vectors()
        
        ghat = fe.Constant(-jhat)
        
        return 1./Pr*(Ra_T*T + Ra_C*C)*ghat
        
    def init_time_discrete_terms(self):
        """ Implicit Euler finite difference scheme """
        p, u, T, C = fe.split(self.solution)
        
        p_n, u_n, T_n, C_n = fe.split(self.initial_values)
        
        Delta_t = self.timestep_size
        
        u_t = (u - u_n)/Delta_t
        
        T_t = (T - T_n)/Delta_t
        
        C_t = (C - C_n)/Delta_t
        
        phi = self.semi_phasefield
        
        phi_t = (phi(T, C) - phi(T_n, C_n))/Delta_t
        
        self.time_discrete_terms = u_t, T_t, C_t, phi_t
        
    def init_weak_form_residual(self):
        """ Weak form from @cite{zimmerman2018monolithic} """
        mu_S = self.solid_dynamic_viscosity
        
        mu_L = self.liquid_dynamic_viscosity
        
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        Sc = self.schmidt_number
        
        gamma = self.pressure_penalty_factor
        
        inner, dot, grad, div, sym = \
            fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
        p, u, T, C = fe.split(self.solution)
        
        u_t, T_t, C_t, phi_t = self.time_discrete_terms
        
        b = self.buoyancy(T, C)
        
        phi = self.semi_phasefield(T, C)
        
        mu = mu_L + (mu_S - mu_L)*phi
        
        psi_p, psi_u, psi_T, psi_C = fe.TestFunctions(self.function_space)
        
        mass = psi_p*div(u)
        
        momentum = dot(psi_u, u_t + grad(u)*u + b) \
            - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), mu*sym(grad(u)))
        
        enthalpy = psi_T*(T_t - 1./Ste*phi_t) \
            + dot(grad(psi_T), 1./Pr*grad(T) - T*u)
        
        concentration = psi_C*((1. - phi)*C_t - C*phi_t) \
            + dot(grad(psi_C), 1./Sc*(1. - phi)*grad(C) - C*u)
            
        stabilization = gamma*psi_p*p
        
        self.weak_form_residual = mass + momentum + enthalpy + stabilization
    
    def plot(self, save = True, show = False):
    
        V = fe.FunctionSpace(
            self.mesh, fe.FiniteElement("P", self.mesh.ufl_cell(), 1))
        
        p, u, T, C = self.solution.split()
        
        phi = fe.interpolate(self.semi_phasefield(T, C), V)
        
        for f, name in zip(
                (self.mesh, p, u, T, C, phi),
                ("\\Omega_h", "p", "\\mathbf{u}", "T", "C", "\\phi")):
            
            fe.plot(f)
            
            plt.axis("square")
        
            plt.xlabel(r"$x$")

            plt.ylabel(r"$y$")

            plt.title(r"$" + name + "$")
            
            if save:
            
                plt.savefig(self.output_prefix + name + ".png")

            if show:
            
                plt.show()
                
            plt.close()
            