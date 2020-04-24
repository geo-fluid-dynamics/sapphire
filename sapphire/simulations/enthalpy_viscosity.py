"""A simulation class using the enthalpy-porosity method.

Use this for convection-coupled melting and solidification.

Dirichlet BC's should not be placed on the pressure.
The returned pressure solution will always have zero mean.

Non-homogeneous Neumann BC's are not implemented for the velocity.
"""
import firedrake as fe
import sapphire.simulation
import sapphire.continuation


def element(cell, degree):
    
    pressure_element = fe.FiniteElement("P", cell, degree[0])
    
    velocity_element = fe.VectorElement("P", cell, degree[1])
    
    temperature_element = fe.FiniteElement("P", cell, degree[2])
    
    return fe.MixedElement(
        pressure_element, velocity_element, temperature_element)
    

erf, sqrt = fe.erf, fe.sqrt

def liquid_volume_fraction(sim, temperature):
    
    T = temperature
    
    T_L = sim.liquidus_temperature
    
    sigma = sim.liquidus_smoothing_factor
    
    return 0.5*(1. + erf((T - T_L)/(sigma*sqrt(2.))))
    
    
def phase_dependent_material_property(solid_to_liquid_ratio):

    a_sl = solid_to_liquid_ratio
    
    def a(phil):
    
        return a_sl + (1. - a_sl)*phil
    
    return a
    
    
def linear_boussinesq_buoyancy(sim, temperature):
    
    T = temperature
    
    Gr = sim.grashof_number
    
    ghat = fe.Constant(-sapphire.simulation.unit_vectors(sim.mesh)[1])
    
    return Gr*T*ghat
    
    
dot, inner, grad, div, sym = fe.dot, fe.inner, fe.grad, fe.div, fe.sym

diff = fe.diff

def strong_residual(sim, solution, buoyancy = linear_boussinesq_buoyancy):
    
    Pr = sim.prandtl_number
    
    Ste = sim.stefan_number
    
    t = sim.time
    
    p, u, T = solution
    
    b = buoyancy(sim = sim, temperature = T)
    
    phil = liquid_volume_fraction(sim = sim, temperature = T)
    
    mu_sl = sim.viscosity_solid_to_liquid_ratio
    
    mu = phase_dependent_material_property(mu_sl)(phil)
    
    rho_sl = sim.density_solid_to_liquid_ratio
    
    c_sl = sim.heat_capacity_solid_to_liquid_ratio
    
    C = phase_dependent_material_property(rho_sl*c_sl)(phil)
    
    k_sl = sim.thermal_conductivity_solid_to_liquid_ratio
    
    k = phase_dependent_material_property(k_sl)(phil)
    
    r_p = div(u)
    
    r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(mu*sym(grad(u))) + b
    
    r_T = diff(C*T, t) + 1./Ste*diff(phil, t) + dot(u, grad(C*T)) \
        - 1./Pr*div(k*grad(T))
    
    return r_p, r_u, r_T
    
    
def time_discrete_terms(sim):
    
    _, u_t, _ = sapphire.simulation.time_discrete_terms(
        solutions = sim.solutions, timestep_size = sim.timestep_size)
    
    temperature_solutions = []
    
    for solution in sim.solutions:
    
        temperature_solutions.append(fe.split(solution)[2])
    
    def phil(T):
    
        return liquid_volume_fraction(sim = sim, temperature = T)
        
    rho_sl = sim.density_solid_to_liquid_ratio
    
    c_sl = sim.heat_capacity_solid_to_liquid_ratio
    
    C = phase_dependent_material_property(rho_sl*c_sl)
    
    CT_t = sapphire.time_discretization.bdf(
        [C(phil(T))*T for T in temperature_solutions],
        timestep_size = sim.timestep_size)
    
    phil_t = sapphire.time_discretization.bdf(
        [phil(T) for T in temperature_solutions],
        timestep_size = sim.timestep_size)
    
    return u_t, CT_t, phil_t
    

def mass(sim, solution):
    
    _, u, _ = fe.split(solution)
    
    psi_p, _, _ = fe.TestFunctions(solution.function_space())
    
    div = fe.div
    
    return psi_p*div(u)
    
    
def momentum(sim, solution, buoyancy = linear_boussinesq_buoyancy):
    
    p, u, T = fe.split(solution)
    
    u_t, _, _ = time_discrete_terms(sim = sim)
    
    b = buoyancy(sim = sim, temperature = T)
    
    phil = liquid_volume_fraction(sim = sim, temperature = T)
    
    mu_sl = sim.viscosity_solid_to_liquid_ratio
    
    mu = phase_dependent_material_property(mu_sl)(phil)
    
    _, psi_u, _ = fe.TestFunctions(solution.function_space())
    
    return dot(psi_u, u_t + grad(u)*u + b) \
        - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), mu*sym(grad(u)))
    
    
def energy(sim, solution):
    
    Pr = sim.prandtl_number
    
    Ste = sim.stefan_number
    
    _, u, T = fe.split(solution)
    
    phil = liquid_volume_fraction(sim = sim, temperature = T)
    
    rho_sl = sim.density_solid_to_liquid_ratio
    
    c_sl = sim.heat_capacity_solid_to_liquid_ratio
    
    C = phase_dependent_material_property(rho_sl*c_sl)(phil)
    
    k_sl = sim.thermal_conductivity_solid_to_liquid_ratio
    
    k = phase_dependent_material_property(k_sl)(phil)
    
    _, CT_t, phil_t = time_discrete_terms(sim = sim)
    
    _, _, psi_T = fe.TestFunctions(solution.function_space())
    
    return psi_T*(CT_t + 1./Ste*phil_t + dot(u, grad(C*T))) \
        + 1./Pr*dot(grad(psi_T), k*grad(T))


def weak_form_residual(sim, solution):
    
    return sum(
            [r(sim = sim, solution = solution) 
            for r in (mass, momentum, energy)])\
        *fe.dx(degree = sim.quadrature_degree)


default_solver_parameters =  {
    "snes_monitor": None,
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxstep": 1.,
    "snes_linesearch_damping": 1.,
    "snes_atol": 1.e-9,
    "snes_stol": 1.e-9,
    "snes_rtol": 0.,
    "snes_max_it": 24,
    "ksp_type": "preonly", 
    "pc_type": "lu", 
    "pc_factor_mat_solver_type": "mumps",
    "mat_type": "aij"}


def nullspace(sim):
    """Inform solver that pressure solution is not unique.
    
    It is only defined up to adding an arbitrary constant.
    """
    W = sim.function_space
    
    return fe.MixedVectorSpaceBasis(
        W, [fe.VectorSpaceBasis(constant=True), W.sub(1), W.sub(2)])


class Simulation(sapphire.simulation.Simulation):
    
    def __init__(
            self, 
            *args, 
            mesh, 
            element_degree = (1, 2, 2), 
            grashof_number = 1.,
            prandtl_number = 1.,
            stefan_number = 1.,
            liquidus_temperature = 0.,
            density_solid_to_liquid_ratio = 1.,
            heat_capacity_solid_to_liquid_ratio = 1.,
            thermal_conductivity_solid_to_liquid_ratio = 1.,
            viscosity_solid_to_liquid_ratio = 1.e8,
            liquidus_smoothing_factor = 0.01,
            solver_parameters = default_solver_parameters,
            **kwargs):
            
        self.grashof_number = fe.Constant(grashof_number)
        
        self.prandtl_number = fe.Constant(prandtl_number)
        
        self.stefan_number = fe.Constant(stefan_number)
        
        self.liquidus_temperature = fe.Constant(liquidus_temperature)
        
        self.density_solid_to_liquid_ratio = fe.Constant(
            density_solid_to_liquid_ratio)
        
        self.heat_capacity_solid_to_liquid_ratio = fe.Constant(
            heat_capacity_solid_to_liquid_ratio)
        
        self.thermal_conductivity_solid_to_liquid_ratio = fe.Constant(
            thermal_conductivity_solid_to_liquid_ratio)
        
        self.viscosity_solid_to_liquid_ratio = fe.Constant(
            viscosity_solid_to_liquid_ratio)
        
        self.liquidus_smoothing_factor = fe.Constant(
            liquidus_smoothing_factor)
        
        self.smoothing_sequence = None
        
        if "weak_form_residual" not in kwargs:
        
            kwargs["weak_form_residual"] = weak_form_residual
        
        if "time_stencil_size" not in kwargs:
        
            kwargs["time_stencil_size"] = 3  # BDF2
            
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            solver_parameters = solver_parameters,
            nullspace = nullspace,
            **kwargs)
            
    def solve(self) -> fe.Function:
        
        self.solution = super().solve()
        
        p, u, T = self.solution.split()
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        mean_pressure = fe.assemble(p*dx)
        
        p = p.assign(p - mean_pressure)
        
        return self.solution
        
    def solve_with_over_regularization(self):
        
        return sapphire.continuation.solve_with_over_regularization(
            solve = self.solve,
            solution = self.solution,
            regularization_parameter = self.liquidus_smoothing_factor)
            
    def solve_with_bounded_regularization_sequence(self):
        
        return sapphire.continuation.\
            solve_with_bounded_regularization_sequence(
                solve = self.solve,
                solution = self.solution,
                backup_solution = self.backup_solution,
                regularization_parameter = self.liquidus_smoothing_factor,
                initial_regularization_sequence = self.smoothing_sequence)
                
    def solve_with_auto_smoothing(self):
        
        sigma = self.liquidus_smoothing_factor.__float__()
        
        if self.smoothing_sequence is None:
        
            given_smoothing_sequence = False
            
        else:
        
            given_smoothing_sequence = True
        
        
        if not given_smoothing_sequence:
            # Find an over-regularization that works.
            self.solution, sigma_max = self.solve_with_over_regularization()
            
            if sigma_max == sigma:
                # No over-regularization was necessary.
                return self.solution
                
            else:
                # A working over-regularization was found, which becomes
                # the upper bound of the sequence.
                self.smoothing_sequence = (sigma_max, sigma)
                
                
        # At this point, either a smoothing sequence has been provided,
        # or a working upper bound has been found.
        # Next, a viable sequence will be sought.
        try:
            
            self.solution, self.smoothing_sequence = \
                self.solve_with_bounded_regularization_sequence()
                
        except fe.exceptions.ConvergenceError as error: 
            
            if given_smoothing_sequence:
                # Try one more time without using the given sequence.
                # This is sometimes useful after solving some time steps
                # with a previously successful regularization sequence
                # that is not working for a new time step.
                self.solution = self.solution.assign(self.solutions[1])
                
                self.solution, smax = self.solve_with_over_regularization()
                
                self.smoothing_sequence = (smax, sigma)
                
                self.solution, self.smoothing_sequence = \
                    self.solve_with_bounded_regularization_sequence()
                    
            else:
                
                raise error
                
                
        # For debugging purposes, ensure that the problem was solved with the 
        # correct regularization and that the simulation's attribute for this
        # has been set to the correct value before returning.
        assert(self.liquidus_smoothing_factor.__float__() == sigma)
        
        return self.solution
        
    def kwargs_for_writeplots(self):
        
        p, u, T = self.solution.split()
        
        phil = fe.interpolate(liquid_volume_fraction(
            sim = self, temperature = T), T.function_space())
        
        return {
            "fields": (p, u, T, phil),
            "labels": ("p", "\\mathbf{u}", "T", "\\phi_l"),
            "names": ("p", "u", "T", "phil"),
            "plotfuns": (fe.tripcolor, fe.quiver, fe.tripcolor, fe.tripcolor)}
            
    def run(self, *args, **kwargs):
        
        return super().run(*args,
            solve = self.solve_with_auto_smoothing,
            **kwargs)
           
    def postprocess(self):
        
        p, u, T = self.solution.split()
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        div = fe.div
        
        self.velocity_divergence = fe.assemble(div(u)*dx)
        
        phil = liquid_volume_fraction(sim = self, temperature = T)
        
        self.liquid_area = fe.assemble(phil*dx)
        
        return self
