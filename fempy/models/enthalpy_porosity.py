""" A enthalpy-porosity model class for convection-coupled phase-change """
import firedrake as fe
import fempy.unsteady_model
import matplotlib.pyplot as plt
import fempy.autosmooth


class Model(fempy.unsteady_model.Model):
    
    def __init__(self):
    
        self.liquid_dynamic_viscosity = fe.Constant(1.)
        
        self.solid_dynamic_viscosity = fe.Constant(1.e8)
        
        self.grashof_number = fe.Constant(1.)
        
        self.prandtl_number = fe.Constant(1.)
        
        self.stefan_number = fe.Constant(1.)
        
        self.pressure_penalty_factor = fe.Constant(1.e-7)
        
        self.liquidus_temperature = fe.Constant(0.)
        
        self.smoothing = fe.Constant(1./256.)
        
        self.smoothing_sequence = None
        
        self.autosmooth_enable = True
        
        self.autosmooth_maxval = 4.
        
        self.autosmooth_firstval = 1./4.
        
        self.autosmooth_maxcount = 32
        
        super().__init__()
        
        self.backup_solution = fe.Function(self.solution)
        
    def init_element(self):
    
        self.element = fe.MixedElement(
            fe.FiniteElement("P", self.mesh.ufl_cell(), 1),
            fe.VectorElement("P", self.mesh.ufl_cell(), 2),
            fe.FiniteElement("P", self.mesh.ufl_cell(), 1))
    
    def porosity(self, T):
        """ Regularization from @cite{zimmerman2018monolithic} """
        T_L = self.liquidus_temperature
        
        s = self.smoothing
        
        tanh = fe.tanh
        
        return 0.5*(1. + tanh((T - T_L)/s))
        
    def buoyancy(self, T):
        """ Boussinesq buoyancy """
        _, _, T = fe.split(self.solution)
        
        Gr = self.grashof_number
        
        _, jhat = self.unit_vectors()
        
        ghat = fe.Constant(-jhat)
        
        return Gr*T*ghat
        
    def viscosity(self, T):
        
        mu_s = self.solid_dynamic_viscosity
        
        mu_l = self.liquid_dynamic_viscosity
        
        phil = self.porosity(T)
        
        return mu_s + (mu_l - mu_s)*phil
        
    def init_time_discrete_terms(self):
        """ Implicit Euler finite difference scheme """
        p, u, T = fe.split(self.solution)
        
        p_n, u_n, T_n = fe.split(self.initial_values)
        
        Delta_t = self.timestep_size
        
        u_t = (u - u_n)/Delta_t
        
        T_t = (T - T_n)/Delta_t
        
        phil = self.porosity
        
        phil_t = (phil(T) - phil(T_n))/Delta_t
        
        self.time_discrete_terms = u_t, T_t, phil_t
        
    def mass(self):
        
        p, u, _ = fe.split(self.solution)
        
        psi_p, _, _ = fe.TestFunctions(self.function_space)
        
        div = fe.div
        
        mass = psi_p*div(u)
        
        gamma = self.pressure_penalty_factor
        
        stabilization = gamma*psi_p*p
        
        return mass + stabilization
        
    def momentum(self):
        
        inner, dot, grad, div, sym = \
            fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
        p, u, T = fe.split(self.solution)
        
        u_t, _, _ = self.time_discrete_terms
        
        b = self.buoyancy(T)
        
        mu = self.viscosity(T)
        
        _, psi_u, _ = fe.TestFunctions(self.function_space)
        
        return dot(psi_u, u_t + grad(u)*u + b) \
            - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), mu*sym(grad(u)))
        
    def enthalpy(self):
        
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        _, u, T = fe.split(self.solution)
        
        _, T_t, phil_t = self.time_discrete_terms
        
        _, _, psi_T = fe.TestFunctions(self.function_space)
        
        dot, grad = fe.dot, fe.grad
        
        return psi_T*(T_t + 1./Ste*phil_t) \
            + dot(grad(psi_T), 1./Pr*grad(T) - T*u)
        
    def init_weak_form_residual(self):
        """ Weak form from @cite{zimmerman2018monolithic} """
        mass = self.mass()
        
        momentum = self.momentum()
        
        enthalpy = self.enthalpy()
        
        self.weak_form_residual = mass + momentum + enthalpy

    def init_integration_measure(self):

        self.integration_measure = fe.dx(degree = 4)
        
    def solve(self):
        
        if self.autosmooth_enable:
            
            fempy.autosmooth.solve(self,
                firstval = self.autosmooth_firstval,
                maxval = self.autosmooth_maxval,
                maxcount = self.autosmooth_maxcount)
            
        elif self.smoothing_sequence == None:
        
            self.solver.solve()
           
        else:
        
            assert(self.smoothing.__float__()
                == self.smoothing_sequence[-1])
        
            for s in self.smoothing_sequence:
            
                self.smoothing.assign(s)
                
                self.solver.solve()
                
                if not self.quiet:
                    
                    print("Solved with s = " + str(s))
    
    def plot(self):
    
        V = fe.FunctionSpace(
            self.mesh, fe.FiniteElement("P", self.mesh.ufl_cell(), 1))
        
        p, u, T = self.solution.split()
        
        phil = fe.interpolate(self.porosity(T), V)
        
        timestr = str(self.time.__float__())
        
        for f, label, filename in zip(
                (self.mesh, p, u, T, phil),
                ("\\Omega_h", "p", "\\mathbf{u}", "T", "\\phi_l"),
                ("mesh", "p", "u", "T", "phil")):
            
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
            
        p_np1, u_np1, T_np1 = fe.split(self.solution)
        
        p_n, u_n, T_n = fe.split(self.initial_values[0])
        
        p_nm1, u_nm1, T_nm1 = fe.split(self.initial_values[1])
        
        u_t = bdf2(u_np1, u_n, u_nm1)
        
        T_t = bdf2(T_np1, T_n, T_nm1)
        
        phil = self.porosity
        
        phil_t = bdf2(phil(T_np1), phil(T_n), phil(T_nm1))
        
        self.time_discrete_terms = u_t, T_t, phil_t
        