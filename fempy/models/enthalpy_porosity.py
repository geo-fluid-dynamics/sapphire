""" A enthalpy-porosity model class for convection-coupled phase-change """
import firedrake as fe
import fempy.unsteady_model
import matplotlib.pyplot as plt
import fempy.autosmooth


class Model(fempy.unsteady_model.Model):
    
    def __init__(self, quadrature_degree, spatial_order, temporal_order):
        
        self.grashof_number = fe.Constant(1.)
        
        self.prandtl_number = fe.Constant(1.)
        
        self.stefan_number = fe.Constant(1.)
        
        self.pressure_penalty_factor = fe.Constant(1.e-7)
        
        self.liquidus_temperature = fe.Constant(0.)
        
        self.solid_velocity_relaxation_factor = fe.Constant(1.e-12)
        
        self.smoothing = fe.Constant(1./256.)
        
        self.smoothing_sequence = None
        
        self.autosmooth_enable = True
        
        self.autosmooth_maxval = 4.
        
        self.autosmooth_firstval = 1./4.
        
        self.autosmooth_maxcount = 32
        
        super().__init__(
            quadrature_degree = quadrature_degree,
            spatial_order = spatial_order,
            temporal_order = temporal_order)
        
        self.backup_solution = fe.Function(self.solution)
        
        # Initialize some attributes to be reported
        self.liquid_area = None
        
    def init_element(self):
        
        rx = self.spatial_order
        
        self.element = fe.MixedElement(
            fe.FiniteElement("P", self.mesh.ufl_cell(), rx - 1),
            fe.VectorElement("P", self.mesh.ufl_cell(), rx),
            fe.FiniteElement("P", self.mesh.ufl_cell(), rx - 1))
    
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
        
    def solid_velocity_relaxation(self, T):
        
        phil = self.porosity(T)
        
        phis = 1. - phil
        
        tau = self.solid_velocity_relaxation_factor
        
        return 1./tau*phis
        
    def init_time_discrete_terms(self):
        
        super().init_time_discrete_terms()
        
        temperature_solutions = []
        
        for solution in self.solutions:
        
            temperature_solutions.append(fe.split(solution)[2])
        
        phil_t = fempy.time_discretization.bdf(
            [self.porosity(T) for T in temperature_solutions],
            order = self.temporal_order,
            timestep_size = self.timestep_size)
        
        self.time_discrete_terms.append(phil_t)
        
    def mass(self):
        
        p, u, _ = fe.split(self.solution)
        
        psi_p, _, _ = fe.TestFunctions(self.function_space)
        
        div = fe.div
        
        mass = psi_p*div(u)
        
        gamma = self.pressure_penalty_factor
        
        stabilization = gamma*psi_p*p
        
        return mass + stabilization
        
    def momentum(self):
        
        p, u, T = fe.split(self.solution)
        
        _, u_t, _, _ = self.time_discrete_terms
        
        b = self.buoyancy(T)
        
        d = self.solid_velocity_relaxation(T)
        
        _, psi_u, _ = fe.TestFunctions(self.function_space)
        
        inner, dot, grad, div, sym = \
            fe.inner, fe.dot, fe.grad, fe.div, fe.sym
            
        return dot(psi_u, u_t + grad(u)*u + b + d*u) \
            - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), sym(grad(u)))
        
    def enthalpy(self):
        
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        _, u, T = fe.split(self.solution)
        
        _, _, T_t, phil_t = self.time_discrete_terms
        
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
    
    def solve(self):
        
        if self.autosmooth_enable:
            
            fempy.autosmooth.solve(self,
                firstval = self.autosmooth_firstval,
                maxval = self.autosmooth_maxval,
                maxcount = self.autosmooth_maxcount)
            
        elif self.smoothing_sequence == None:
        
            super().solve()
           
        else:
        
            assert(self.smoothing.__float__()
                == self.smoothing_sequence[-1])
        
            for s in self.smoothing_sequence:
            
                self.smoothing.assign(s)
                
                super().solve()
                
                if not self.quiet:
                    
                    print("Solved with s = " + str(s))
    
    def report(self, write_header):
        
        p, u, T = self.solution.split()
    
        phil = self.porosity(T)
        
        self.liquid_area = fe.assemble(phil*self.integration_measure)
        
        super().report(write_header = write_header)
    
    def plot(self):
    
        V = fe.FunctionSpace(
            self.mesh, fe.FiniteElement("P", self.mesh.ufl_cell(), 1))
        
        p, u, T = self.solution.split()
        
        phil = fe.interpolate(self.porosity(T), V)
        
        timestr = str(self.time.__float__())
        
        for f, label, filename in zip(
                (p, u, T, phil),
                ("p", "\\mathbf{u}", "T", "\\phi_l"),
                ("p", "u", "T", "phil")):
            
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
