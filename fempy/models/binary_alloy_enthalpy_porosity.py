""" A enthalpy-porosity model class for convection-coupled phase-change
of binary alloys
"""
import firedrake as fe
import fempy.models.enthalpy_porosity
import matplotlib.pyplot as plt
import pathlib

    
class Model(fempy.models.enthalpy_porosity.Model):
    
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
    
    def concentration_dependent_liquidus_temperature(self, Cl):
    
        T_m = self.pure_liquidus_temperature
        
        m_L = self.liquidus_slope
        
        return T_m + m_L*Cl
    
    def porosity(self, T, Cl):
        """ Regularization from @cite{zimmerman2018monolithic} """
        T_L = self.concentration_dependent_liquidus_temperature
        
        s = self.latent_heat_smoothing
        
        tanh = fe.tanh
        
        return 0.5*(1. + tanh((T - T_L(Cl))/s))
        
    def buoyancy(self, T, Cl):
        """ Boussinesq buoyancy """
        Ra_T = self.temperature_rayleigh_number
        
        Ra_C = self.concentration_rayleigh_number
        
        Pr = self.prandtl_number
        
        ihat, jhat = self.unit_vectors()
        
        ghat = fe.Constant(-jhat)
        
        return 1./Pr*(Ra_T*T + Ra_C*Cl)*ghat
        
    def init_time_discrete_terms(self):
        """ Implicit Euler finite difference scheme """
        p, u, T, C = fe.split(self.solution)
        
        p_n, u_n, T_n, C_n = fe.split(self.initial_values)
        
        Delta_t = self.timestep_size
        
        u_t = (u - u_n)/Delta_t
        
        T_t = (T - T_n)/Delta_t
        
        C_t = (C - C_n)/Delta_t
        
        phil = self.porosity
        
        phil_t = (phil(T, C) - phil(T_n, C_n))/Delta_t
        
        self.time_discrete_terms = u_t, T_t, C_t, phil_t
        
    def init_weak_form_residual(self):
        """ Weak form from @cite{zimmerman2018monolithic} """
        mu_s = self.solid_dynamic_viscosity
        
        mu_l = self.liquid_dynamic_viscosity
        
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        Sc = self.schmidt_number
        
        gamma = self.pressure_penalty_factor
        
        inner, dot, grad, div, sym = \
            fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
        p, u, T, Cl = fe.split(self.solution)
        
        u_t, T_t, Cl_t, phil_t = self.time_discrete_terms
        
        b = self.buoyancy(T, Cl)
        
        phil = self.porosity(T, Cl)
        
        mu = mu_s + (mu_l - mu_s)*phil
        
        psi_p, psi_u, psi_T, psi_C = fe.TestFunctions(self.function_space)
        
        mass = psi_p*div(u)
        
        momentum = dot(psi_u, u_t + grad(u)*u + b) \
            - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), mu*sym(grad(u)))
        
        enthalpy = psi_T*(T_t + 1./Ste*phil_t) \
            + dot(grad(psi_T), 1./Pr*grad(T) - T*u)
        
        concentration = psi_C*(phil*Cl_t + Cl*phil_t) \
            + dot(grad(psi_C), 1./Sc*phil*grad(Cl) - Cl*u)
            
        stabilization = gamma*psi_p*p
        
        self.weak_form_residual = mass + momentum + enthalpy + concentration \
            + stabilization
    
    def plot(self):
    
        V = fe.FunctionSpace(
            self.mesh, fe.FiniteElement("P", self.mesh.ufl_cell(), 1))
        
        p, u, T, Cl = self.solution.split()
        
        _phil = self.porosity(T, Cl)
        
        phil = fe.interpolate(_phil, V)
        
        C = fe.interpolate(_phil*Cl, V)
        
        timestr = str(self.time.__float__())
        
        for f, label, filename in zip(
                (self.mesh, p, u, T, C, phil),
                ("\\Omega_h", "p", "\\mathbf{u}", "T", "C", "\\phi_l"),
                ("mesh", "p", "u", "T", "Cl", "phil")):
            
            fe.plot(f)
            
            plt.axis("square")
        
            plt.xlabel(r"$x$")

            plt.ylabel(r"$y$")

            plt.title(r"$" + label + 
                ", t = " + timestr + "$")
            
            self.output_directory_path.mkdir(
                parents = True, exist_ok = True)
        
            filepath = self.output_directory_path.joinpath(filename + 
                "_t" + timestr.replace(".", "p")).with_suffix(".png")
            
            print("Writing plot to " + str(filepath))
            
            plt.savefig(str(filepath))
            
            plt.close()
            
            
class ModelWithBDF2(Model):
    
    def init_time_discrete_terms(self):
        
        Delta_t = self.timestep_size
        
        def bdf2(u_np1, u_n, u_nm1):
        
            return (3.*u_np1 - 4.*u_n + u_nm1)/(2.*Delta_t)
            
        p_np1, u_np1, T_np1, C_np1 = fe.split(self.solution)
        
        p_n, u_n, T_n, C_n = fe.split(self.initial_values[0])
        
        p_nm1, u_nm1, T_nm1, C_nm1 = fe.split(self.initial_values[1])
        
        u_t = bdf2(u_np1, u_n, u_nm1)
        
        T_t = bdf2(T_np1, T_n, T_nm1)
        
        C_t = bdf2(C_np1, C_n, C_nm1)
        
        phil = self.porosity
        
        phil_t = bdf2(
            phil(T_np1, C_np1), phil(T_n, C_n), phil(T_nm1, C_nm1))
        
        self.time_discrete_terms = u_t, T_t, C_t, phil_t
        