"""A sim class for convection-coupled melting and solidification in enthalpy form"""
import firedrake as fe
import sapphire.simulation
import sapphire.continuation


def element(cell, degree):
    
    if type(degree) is type(1):
    
        degree = (degree,)*3
        
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
    
    
def solid_velocity_relaxation(sim, temperature):
    
    T = temperature
    
    phil = liquid_volume_fraction(sim = sim, temperature = T)
    
    phis = 1. - phil
    
    tau = sim.solid_velocity_relaxation_factor
    
    return 1./tau*phis
    

dot, inner, grad, div, sym = fe.dot, fe.inner, fe.grad, fe.div, fe.sym

diff = fe.diff

def strong_residual(sim, solution, buoyancy = linear_boussinesq_buoyancy):
    
    Pr = sim.prandtl_number
    
    Ste = sim.stefan_number
    
    t = sim.time
    
    p, u, T = solution
    
    b = buoyancy(sim = sim, temperature = T)
    
    d = solid_velocity_relaxation(sim = sim, temperature = T)
    
    phil = liquid_volume_fraction(sim = sim, temperature = T)
    
    rho_sl = sim.density_solid_to_liquid_ratio
    
    c_sl = sim.heat_capacity_solid_to_liquid_ratio
    
    C = phase_dependent_material_property(rho_sl*c_sl)(phil)
    
    k_sl = sim.thermal_conductivity_solid_to_liquid_ratio
    
    k = phase_dependent_material_property(k_sl)(phil)
    
    r_p = div(u)
    
    r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(sym(grad(u))) + b + d*u
    
    r_T = diff(C*T, t) + 1./Ste*diff(phil, t) + dot(u, grad(C*T)) \
        - 1./Pr*div(k*grad(T))
        
    
    return r_p, r_u, r_T
    
    
def strong_residual_with_pressure_penalty(sim, solution, buoyancy = linear_boussinesq_buoyancy):
    
    r_p, r_u, r_T = strong_residual(sim = sim, solution = solution, buoyancy = buoyancy)
    
    p, _, _ = solution
    
    gamma = sim.pressure_penalty_factor
    
    return r_p + gamma*p, r_u, r_T 
    
    
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
    
    mass = psi_p*div(u)
    
    return mass
    
    
def momentum(sim, solution, buoyancy = linear_boussinesq_buoyancy):
    
    p, u, T = fe.split(solution)
    
    u_t, _, _ = time_discrete_terms(sim = sim)
    
    b = buoyancy(sim = sim, temperature = T)
    
    d = solid_velocity_relaxation(sim = sim, temperature = T)
    
    _, psi_u, _ = fe.TestFunctions(solution.function_space())
        
    return dot(psi_u, u_t + grad(u)*u + b + d*u) \
        - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), sym(grad(u)))
    
    
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
        
    
def pressure_penalty(sim, solution):

    p, _, _ = fe.split(solution)
    
    psi_p, _, _ = fe.TestFunctions(solution.function_space())
    
    gamma = sim.pressure_penalty_factor
    
    return gamma*psi_p*p
    
    
def weak_form_residual(sim, solution):
    
    return sum(
            [r(sim = sim, solution = solution) 
            for r in (mass, momentum, energy, pressure_penalty)])\
        *fe.dx(degree = sim.quadrature_degree)

    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(
            self, 
            *args, 
            mesh, 
            element_degree = (1, 2, 2), 
            grashof_number = 1.,
            prandtl_number = 1.,
            stefan_number = 1.,
            pressure_penalty_factor = 1.e-7,
            liquidus_temperature = 0.,
            density_solid_to_liquid_ratio = 1.,
            heat_capacity_solid_to_liquid_ratio = 1.,
            thermal_conductivity_solid_to_liquid_ratio = 1.,
            solid_velocity_relaxation_factor = 1.e-12,
            liquidus_smoothing_factor = 0.01,
            solver_parameters = {
                "snes_type": "newtonls",
                "snes_max_it": 24,
                "snes_monitor": None,
                "snes_rtol": 0.,
                "ksp_type": "preonly", 
                "pc_type": "lu", 
                "mat_type": "aij",
                "pc_factor_mat_solver_type": "mumps"},
            **kwargs):
        
        self.grashof_number = fe.Constant(grashof_number)
        
        self.prandtl_number = fe.Constant(prandtl_number)
        
        self.stefan_number = fe.Constant(stefan_number)
        
        self.pressure_penalty_factor = fe.Constant(pressure_penalty_factor)
        
        self.liquidus_temperature = fe.Constant(liquidus_temperature)
        
        self.density_solid_to_liquid_ratio = fe.Constant(
            density_solid_to_liquid_ratio)
        
        self.heat_capacity_solid_to_liquid_ratio = fe.Constant(
            heat_capacity_solid_to_liquid_ratio)
        
        self.thermal_conductivity_solid_to_liquid_ratio = fe.Constant(
            thermal_conductivity_solid_to_liquid_ratio)
        
        self.solid_velocity_relaxation_factor = fe.Constant(
            solid_velocity_relaxation_factor)
        
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
            **kwargs)
            
    def solve_with_auto_smoothing(self):
        
        s0 = self.liquidus_smoothing_factor.__float__()
        
        def solve_with_over_regularization(self, startval):
        
            return sapphire.continuation.solve_with_over_regularization(
                solve = self.solve,
                solution = self.solution,
                regularization_parameter = self.liquidus_smoothing_factor,
                startval = startval)
        
        def solve_with_bounded_regularization_sequence(self):
        
            return sapphire.continuation.\
                solve_with_bounded_regularization_sequence(
                    solve = self.solve,
                    solution = self.solution,
                    backup_solution = self.backup_solution,
                    regularization_parameter = self.liquidus_smoothing_factor,
                    initial_regularization_sequence = self.smoothing_sequence)
                    
        if self.smoothing_sequence is None:
        
            self.solution, smax = solve_with_over_regularization(
                self, startval = None)
            
            s = self.liquidus_smoothing_factor.__float__()
            
            if s == smax:
            
                self.smoothing_sequence = (s,)
                
            else:
            
                self.smoothing_sequence = (smax, s)
            
        try:
            
            self.solution, self.smoothing_sequence = \
                solve_with_bounded_regularization_sequence(self)
                
        except fe.exceptions.ConvergenceError: 
            # Try one more time.
            self.solution, smax = solve_with_over_regularization(
                self, startval = self.smoothing_sequence[-1])
            
            self.smoothing_sequence = (
                smax, self.liquidus_smoothing_factor.__float__())
            
            self.solution, self.smoothing_sequence = \
                solve_with_bounded_regularization_sequence(self)
               
        assert(self.liquidus_smoothing_factor.__float__() == s0)
        
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
        
        self.mean_pressure = fe.assemble(p*dx)
        
        self.velocity_divergence = fe.assemble(div(u)*dx)
        
        phil = liquid_volume_fraction(sim = self, temperature = T)
        
        self.liquid_area = fe.assemble(phil*dx)
        
        return self
        