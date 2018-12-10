""" A convection-coupled phase-change model class """
import firedrake as fe
import fempy.unsteady_model
import matplotlib.pyplot as plt

    
class Model(fempy.unsteady_model.Model):
    
    def __init__(self):
    
        self.solid_dynamic_viscosity = fe.Constant(1.e8)
        
        self.rayleigh_number = fe.Constant(1.)
        
        self.prandtl_number = fe.Constant(1.)
        
        self.stefan_number = fe.Constant(1.)
        
        self.pressure_penalty_factor = fe.Constant(1.e-7)
        
        self.liquidus_temperature = fe.Constant(0.)
        
        self.phase_interface_smoothing = fe.Constant(1./256.)
        
        self.smoothing_sequence = None
        
        self.autosmooth_enable = True
        
        self.autosmooth_maxval = 4.
        
        self.autosmooth_firstval = 1./4.
        
        self.autosmooth_maxcount = 16
        
        super().__init__()
        
        self.backup_solution = fe.Function(self.solution)
        
    def init_element(self):
    
        self.element = fe.MixedElement(
            fe.FiniteElement("P", self.mesh.ufl_cell(), 1),
            fe.VectorElement("P", self.mesh.ufl_cell(), 2),
            fe.FiniteElement("P", self.mesh.ufl_cell(), 1))
    
    def semi_phasefield(self, T):
        """ Regularization from @cite{zimmerman2018monolithic} """
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
        
    def init_time_discrete_terms(self):
        """ Implicit Euler finite difference scheme """
        p, u, T = fe.split(self.solution)
        
        p_n, u_n, T_n = fe.split(self.initial_values)
        
        Delta_t = self.timestep_size
        
        u_t = (u - u_n)/Delta_t
        
        T_t = (T - T_n)/Delta_t
        
        phi = self.semi_phasefield
        
        phi_t = (phi(T) - phi(T_n))/Delta_t
        
        self.time_discrete_terms = u_t, T_t, phi_t
        
    def init_weak_form_residual(self):
        """ Weak form from @cite{zimmerman2018monolithic} """
        mu_S = self.solid_dynamic_viscosity
        
        mu_L = fe.Constant(1.)
        
        Ra = self.rayleigh_number
        
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        gamma = self.pressure_penalty_factor
        
        inner, dot, grad, div, sym = \
            fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
        p, u, T = fe.split(self.solution)
        
        u_t, T_t, phi_t = self.time_discrete_terms
        
        b = self.buoyancy(T)
        
        phi = self.semi_phasefield(T)
        
        mu = mu_L + (mu_S - mu_L)*phi
        
        psi_p, psi_u, psi_T = fe.TestFunctions(self.function_space)
        
        mass = psi_p*div(u)
        
        momentum = dot(psi_u, u_t + grad(u)*u + b) \
            - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), mu*sym(grad(u)))
        
        enthalpy = psi_T*(T_t - 1./Ste*phi_t) \
            + dot(grad(psi_T), 1./Pr*grad(T) - T*u)
        
        stabilization = gamma*psi_p*p
        
        self.weak_form_residual = mass + momentum + enthalpy + stabilization

    def init_solver(self, solver_parameters = {
            "snes_type": "newtontr",
            "snes_monitor": True,
            "ksp_type": "preonly", 
            "pc_type": "lu", 
            "mat_type": "aij",
            "pc_factor_mat_solver_type": "mumps"}):
        
        super().init_solver(solver_parameters = solver_parameters)
        
    def solve(self, *args, **kwargs):
        
        if self.autosmooth_enable:
            
            self.solve_with_auto_smoothing()
            
        elif self.smoothing_sequence == None:
        
            self.solver.solve()
           
        else:
        
            assert(self.phase_interface_smoothing.__float__()
                == self.smoothing_sequence[-1])
        
            for s in self.smoothing_sequence:
            
                self.phase_interface_smoothing.assign(s)
                
                self.solver.solve()
                
                if not self.quiet:
                    
                    print("Solved with s = " + str(s))
        
    def solve_with_auto_smoothing(self):
            
        if self.smoothing_sequence is None:
        
            smoothing_sequence = (self.phase_interface_smoothing.__float__(),)
            
            while smoothing_sequence[0] < self.autosmooth_firstval:

                smoothing_sequence = (2.*smoothing_sequence[0],) + \
                    smoothing_sequence
            
        else:
        
            smoothing_sequence = self.smoothing_sequence
        
        first_s_to_solve = smoothing_sequence[0]
        
        attempts = range(self.autosmooth_maxcount - len(smoothing_sequence))
        
        solved = False
        
        for attempt in attempts:

            s_start_index = smoothing_sequence.index(first_s_to_solve)
            
            try:
            
                for s in smoothing_sequence[s_start_index:]:
                    
                    self.phase_interface_smoothing.assign(s)
                    
                    self.backup_solution.assign(self.solution)
                    
                    self.solver.solve()
                    
                    if not self.quiet:
                    
                        print("Solved with s = " + str(s))
                    
                solved = True
                
                break
                
            except fe.exceptions.ConvergenceError:  
                
                current_s = self.phase_interface_smoothing.__float__()
                
                ss = smoothing_sequence
                
                if not self.quiet:
                
                    print("Failed to solve with s = " + str(current_s) +
                        " from the sequence " + str(ss))
                
                if attempt == attempts[-1]:
                    
                    break
                
                if current_s >= self.autosmooth_maxval:
                
                    print("Exceeded maximum regularization (s_max = " + 
                        str(self.autosmooth_maxval) + ")")
                    
                    break
                
                index = ss.index(current_s)
                
                if index == 0:
                
                    s_to_insert = 2.*ss[0]
                    
                    new_ss = (s_to_insert,) + ss
                    
                    self.solution.assign(self.initial_values[0])
                
                else:
                
                    s_to_insert = (current_s + ss[index - 1])/2.
                
                    new_ss = ss[:index] + (s_to_insert,) + ss[index:]
                    
                    self.solution.assign(self.backup_solution)
                
                smoothing_sequence = new_ss
                
                if not self.quiet:
                
                    print("Inserted new value of " + str(s_to_insert))
                
                first_s_to_solve = s_to_insert
        
        assert(solved)
        
        assert(self.phase_interface_smoothing.__float__() ==
            smoothing_sequence[-1])
            
        self.smoothing_sequence = smoothing_sequence
        
    def plot(self, save = True, show = False):
    
        V = fe.FunctionSpace(
            self.mesh, fe.FiniteElement("P", self.mesh.ufl_cell(), 1))
        
        p, u, T = self.solution.split()
        
        phi = fe.interpolate(self.semi_phasefield(T), V)
        
        for f, name in zip(
                (self.mesh, p, u, T, phi),
                ("\Omega_h", "p", "\mathbf{u}", "T", "\phi")):
            
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
            