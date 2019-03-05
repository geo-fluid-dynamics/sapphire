""" A enthalpy-porosity model class for convection-coupled phase-change """
import firedrake as fe
import fempy.model
import fempy.continuation


class Model(fempy.model.Model):
    
    def __init__(self, *args, mesh, element_degree, **kwargs):
        
        self.grashof_number = fe.Constant(1.)
        
        self.prandtl_number = fe.Constant(1.)
        
        self.stefan_number = fe.Constant(1.)
        
        self.pressure_penalty_factor = fe.Constant(1.e-7)
        
        self.liquidus_temperature = fe.Constant(0.)
        
        self.heat_capacity_solid_to_liquid_ratio = fe.Constant(1.)
        
        self.thermal_conductivity_solid_to_liquid_ratio = fe.Constant(1.)
        
        self.solid_velocity_relaxation_factor = fe.Constant(1.e-12)
        
        self.smoothing = fe.Constant(1./256.)
        
        self.smoothing_sequence = None
        
        self.save_smoothing_sequence = False
        
        self.autosmooth_enable = True
        
        self.autosmooth_maxval = 4.
        
        self.autosmooth_maxcount = 32
        
        scalar = fe.FiniteElement("P", mesh.ufl_cell(), element_degree)
        
        vector = fe.VectorElement("P", mesh.ufl_cell(), element_degree)
        
        element = fe.MixedElement(scalar, vector, scalar)
            
        super().__init__(*args, mesh = mesh, element = element, **kwargs)
        
        self.backup_solution = fe.Function(self.solution)
        
        self.liquid_area = None
        
    def porosity(self, T):
        """ Regularization from @cite{zimmerman2018monolithic} """
        T_L = self.liquidus_temperature
        
        s = self.smoothing
        
        tanh = fe.tanh
        
        return 0.5*(1. + tanh((T - T_L)/s))
        
    def phase_dependent_material_property(self, solid_to_liquid_ratio):
    
        a_s = solid_to_liquid_ratio
        
        def a(phil):
        
            return a_s + (1. - a_s)*phil
        
        return a
        
    def buoyancy(self, T):
        """ Boussinesq buoyancy """
        _, _, T = fe.split(self.solution)
        
        Gr = self.grashof_number
        
        ghat = fe.Constant(-self.unit_vectors()[1])
        
        return Gr*T*ghat
        
    def solid_velocity_relaxation(self, T):
        
        phil = self.porosity(T)
        
        phis = 1. - phil
        
        tau = self.solid_velocity_relaxation_factor
        
        return 1./tau*phis
        
    def strong_form_residual(self, solution):
        """ This is the strong form of the PDE 
        approximated by the weak form.
        """
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        t = self.time
        
        grad, dot, div, sym, diff = fe.grad, fe.dot, fe.div, fe.sym, fe.diff
        
        p, u, T = solution
        
        b = self.buoyancy(T)
        
        d = self.solid_velocity_relaxation(T)
        
        phil = self.porosity(T)
        
        cp = self.phase_dependent_material_property(
            self.heat_capacity_solid_to_liquid_ratio)(phil)
        
        k = self.phase_dependent_material_property(
            self.thermal_conductivity_solid_to_liquid_ratio)(phil)
            
        r_p = div(u)
        
        r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(sym(grad(u))) \
            + b + d*u
        
        r_T = diff(cp*T, t) + dot(u, grad(cp*T)) \
            - 1./Pr*div(k*grad(T)) + 1./Ste*diff(cp*phil, t)
        
        return r_p, r_u, r_T
        
    def init_time_discrete_terms(self):
        
        super().init_time_discrete_terms()
        
        temperature_solutions = []
        
        for solution in self.solutions:
        
            temperature_solutions.append(fe.split(solution)[2])
        
        phil = self.porosity
        
        cp = self.phase_dependent_material_property(
            self.heat_capacity_solid_to_liquid_ratio)
            
        cpT_t = fempy.time_discretization.bdf(
            [cp(phil(T))*T for T in temperature_solutions],
            order = self.time_stencil_size - 1,
            timestep_size = self.timestep_size)
        
        cpphil_t = fempy.time_discretization.bdf(
            [cp(phil(T))*self.porosity(T) for T in temperature_solutions],
            order = self.time_stencil_size - 1,
            timestep_size = self.timestep_size)
        
        for w_i_t in (cpT_t, cpphil_t):
        
            self.time_discrete_terms.append(w_i_t)
    
    def mass(self):
        
        _, u, _ = fe.split(self.solution)
        
        psi_p, _, _ = fe.TestFunctions(self.function_space)
        
        div = fe.div
        
        mass = psi_p*div(u)
        
        return mass
        
    def momentum(self):
        
        p, u, T = fe.split(self.solution)
        
        _, u_t, _, _, _ = self.time_discrete_terms
        
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
        
        phil = self.porosity(T)
        
        cp = self.phase_dependent_material_property(
            self.heat_capacity_solid_to_liquid_ratio)(phil)
        
        k = self.phase_dependent_material_property(
            self.thermal_conductivity_solid_to_liquid_ratio)(phil)
        
        _, _, _, cpT_t, cpphil_t = self.time_discrete_terms
        
        _, _, psi_T = fe.TestFunctions(self.function_space)
        
        dot, grad = fe.dot, fe.grad
        
        return psi_T*(cpT_t + dot(u, cp*grad(T)) + 1./Ste*cpphil_t) \
            + dot(grad(psi_T), k/Pr*grad(T))
        
    def stabilization(self):
    
        p, _, _ = fe.split(self.solution)
        
        psi_p, _, _ = fe.TestFunctions(self.function_space)
        
        gamma = self.pressure_penalty_factor
        
        return gamma*psi_p*p
        
    def init_weak_form_residual(self):
        """ Weak form from @cite{zimmerman2018monolithic} """
        mass = self.mass()
        
        momentum = self.momentum()
        
        enthalpy = self.enthalpy()
        
        stabilization = self.stabilization()
        
        self.weak_form_residual = mass + momentum + enthalpy + stabilization
    
    def solve(self):
        
        if self.autosmooth_enable:
            
            smoothing_sequence = fempy.continuation.solve(
                model = self,
                solver = self.solver,
                continuation_parameter = self.smoothing,
                continuation_sequence = self.smoothing_sequence,
                leftval = self.autosmooth_maxval,
                rightval = self.smoothing.__float__(),
                startleft = True,
                maxcount = self.autosmooth_maxcount)
                
            if self.save_smoothing_sequence:
            
                self.smoothing_sequence = smoothing_sequence
            
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
    
    def postprocess(self):
        
        p, u, T = self.solution.split()
    
        phil = self.porosity(T)
        
        self.liquid_area = fe.assemble(phil*self.integration_measure)
        
    def plotvars(self):
    
        V = fe.FunctionSpace(
            self.mesh, fe.FiniteElement("P", self.mesh.ufl_cell(), 1))
        
        p, u, T = self.solution.split()
        
        phil = fe.interpolate(self.porosity(T), V)
        
        return (p, u, T, phil), \
            ("p", "\\mathbf{u}", "T", "\\phi_l"), \
            ("p", "u", "T", "phil")
