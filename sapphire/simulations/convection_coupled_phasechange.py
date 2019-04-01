"""A sim class for convection-coupled melting and solidification in enthalpy form"""
import firedrake as fe
import sapphire.simulation
import sapphire.continuation


def element(cell, degree):

    scalar = fe.FiniteElement("P", cell, degree)
    
    vector = fe.VectorElement("P", cell, degree)
    
    return fe.MixedElement(scalar, vector, scalar)
    

erf, sqrt = fe.erf, fe.sqrt

def liquid_volume_fraction(sim, temperature):
    
    T = temperature
    
    T_L = sim.liquidus_temperature
    
    sigma = sim.smoothing
    
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
        
    
def stabilization(sim, solution):

    p, _, _ = fe.split(solution)
    
    psi_p, _, _ = fe.TestFunctions(solution.function_space())
    
    gamma = sim.pressure_penalty_factor
    
    return gamma*psi_p*p
    
    
def variational_form_residual(sim, solution):
    
    return sum(
            [r(sim = sim, solution = solution) 
            for r in (mass, momentum, energy, stabilization)])\
        *fe.dx(degree = sim.quadrature_degree)

    
def plotvars(sim, solution = None):
    
    if solution is None:
    
        solution = sim.solution
        
    V = fe.FunctionSpace(
        solution.function_space().mesh(),
        fe.FiniteElement("P", sim.mesh.ufl_cell(), 1))
    
    p, u, T = solution.split()
    
    phil = fe.interpolate(liquid_volume_fraction(
        sim = sim, temperature = T), V)
    
    return (p, u, T, phil), \
        ("p", "\\mathbf{u}", "T", "\\phi_l"), \
        ("p", "u", "T", "phil")
    
    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args, mesh, element_degree, **kwargs):
        
        self.grashof_number = fe.Constant(1.)
        
        self.prandtl_number = fe.Constant(1.)
        
        self.stefan_number = fe.Constant(1.)
        
        self.pressure_penalty_factor = fe.Constant(1.e-7)
        
        self.liquidus_temperature = fe.Constant(0.)
        
        self.density_solid_to_liquid_ratio = fe.Constant(1.)
        
        self.heat_capacity_solid_to_liquid_ratio = fe.Constant(1.)
        
        self.thermal_conductivity_solid_to_liquid_ratio = fe.Constant(1.)
        
        self.solid_velocity_relaxation_factor = fe.Constant(1.e-12)
        
        self.smoothing = fe.Constant(1./256.)
        
        self.smoothing_sequence = None
        
        if "variational_form_residual" not in kwargs:
        
            kwargs["variational_form_residual"] = variational_form_residual
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            **kwargs)
            
    def solve(self, *args, **kwargs):
    
        return super().solve(*args,
            parameters = {
                "snes_type": "newtonls",
                "snes_max_it": 24,
                "snes_monitor": True,
                "snes_converged_reason": False,
                "ksp_type": "preonly", 
                "pc_type": "lu", 
                "mat_type": "aij",
                "pc_factor_mat_solver_type": "mumps"},
            **kwargs)
            
    def solve_with_auto_smoothing(self):
        
        s0 = self.smoothing.__float__()
        
        def solve_with_over_regularization(self, startval):
        
            return sapphire.continuation.solve_with_over_regularization(
                solve = self.solve,
                solution = self.solution,
                regularization_parameter = self.smoothing,
                startval = startval)
        
        def solve_with_bounded_regularization_sequence(self):
        
            return sapphire.continuation.\
                solve_with_bounded_regularization_sequence(
                    solve = self.solve,
                    solution = self.solution,
                    backup_solution = self.backup_solution,
                    regularization_parameter = self.smoothing,
                    initial_regularization_sequence = self.smoothing_sequence)
                    
        if self.smoothing_sequence is None:
        
            self.solution, smax = solve_with_over_regularization(
                self, startval = None)
            
            self.smoothing_sequence = (smax, self.smoothing.__float__())
            
        try:
            
            self.solution, self.smoothing_sequence = \
                solve_with_bounded_regularization_sequence(self)
                
        except fe.exceptions.ConvergenceError: 
            # Try one more time.
            self.solution, smax = solve_with_over_regularization(
                self, startval = self.smoothing_sequence[-1])
            
            self.smoothing_sequence = (smax, self.smoothing.__float__())
            
            self.solution, self.smoothing_sequence = \
                solve_with_bounded_regularization_sequence(self)
               
        assert(self.smoothing.__float__() == s0)
               
        return self.solution
    
    def run(self, *args, **kwargs):
        
        return super().run(*args,
            solve = self.solve_with_auto_smoothing,
            **kwargs)
           
    def postprocess(self):
        
        _, _, T = self.solution.split()
        
        phil = liquid_volume_fraction(sim = self, temperature = T)
        
        self.liquid_area = fe.assemble(phil*fe.dx)
        
        return self
        
    def write_outputs(self, *args, **kwargs):
        
        super().write_outputs(*args, plotvars = plotvars, **kwargs)
        