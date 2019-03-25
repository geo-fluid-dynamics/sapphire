"""A enthalpy model class for convection-coupled melting and solidification"""
import firedrake as fe
import fempy.model
import fempy.continuation


def element(cell, degree):

    scalar = fe.FiniteElement("P", cell, degree)
    
    vector = fe.VectorElement("P", cell, degree)
    
    return fe.MixedElement(scalar, vector, scalar)
    

erf, sqrt = fe.erf, fe.sqrt

def liquid_volume_fraction(model, temperature):
    
    T = temperature
    
    T_L = model.liquidus_temperature
    
    sigma = model.smoothing
    
    return 0.5*(1. + erf((T - T_L)/(sigma*sqrt(2.))))
    
    
def phase_dependent_material_property(solid_to_liquid_ratio):

    a_sl = solid_to_liquid_ratio
    
    def a(phil):
    
        return a_sl + (1. - a_sl)*phil
    
    return a
    
    
def linear_boussinesq_buoyancy(model, temperature):
    
    T = temperature
    
    Gr = model.grashof_number
    
    ghat = fe.Constant(-fempy.model.unit_vectors(model.mesh)[1])
    
    return Gr*T*ghat
    
    
def solid_velocity_relaxation(model, temperature):
    
    T = temperature
    
    phil = liquid_volume_fraction(model = model, temperature = T)
    
    phis = 1. - phil
    
    tau = model.solid_velocity_relaxation_factor
    
    return 1./tau*phis
    

dot, inner, grad, div, sym = fe.dot, fe.inner, fe.grad, fe.div, fe.sym

diff = fe.diff

def strong_residual(model, solution, buoyancy = linear_boussinesq_buoyancy):
    
    Pr = model.prandtl_number
    
    Ste = model.stefan_number
    
    t = model.time
    
    p, u, T = solution
    
    b = buoyancy(model = model, temperature = T)
    
    d = solid_velocity_relaxation(model = model, temperature = T)
    
    phil = liquid_volume_fraction(model = model, temperature = T)
    
    rho_sl = model.density_solid_to_liquid_ratio
    
    c_sl = model.heat_capacity_solid_to_liquid_ratio
    
    C = phase_dependent_material_property(rho_sl*c_sl)(phil)
    
    k_sl = model.thermal_conductivity_solid_to_liquid_ratio
    
    k = phase_dependent_material_property(k_sl)(phil)
    
    r_p = div(u)
    
    r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(sym(grad(u))) + b + d*u
    
    r_T = diff(C*T, t) + 1./Ste*diff(phil, t) + dot(u, grad(C*T)) \
        - 1./Pr*div(k*grad(T))
        
    
    return r_p, r_u, r_T
    
    
def time_discrete_terms(model):
    
    _, u_t, _ = fempy.model.time_discrete_terms(
        solutions = model.solutions, timestep_size = model.timestep_size)
    
    temperature_solutions = []
    
    for solution in model.solutions:
    
        temperature_solutions.append(fe.split(solution)[2])
    
    def phil(T):
    
        return liquid_volume_fraction(model = model, temperature = T)
        
    rho_sl = model.density_solid_to_liquid_ratio
    
    c_sl = model.heat_capacity_solid_to_liquid_ratio
    
    C = phase_dependent_material_property(rho_sl*c_sl)
    
    CT_t = fempy.time_discretization.bdf(
        [C(phil(T))*T for T in temperature_solutions],
        timestep_size = model.timestep_size)
    
    phil_t = fempy.time_discretization.bdf(
        [phil(T) for T in temperature_solutions],
        timestep_size = model.timestep_size)
    
    return u_t, CT_t, phil_t
    

def mass(model, solution):
    
    _, u, _ = fe.split(solution)
    
    psi_p, _, _ = fe.TestFunctions(solution.function_space())
    
    div = fe.div
    
    mass = psi_p*div(u)
    
    return mass
    
    
def momentum(model, solution, buoyancy = linear_boussinesq_buoyancy):
    
    p, u, T = fe.split(solution)
    
    u_t, _, _ = time_discrete_terms(model = model)
    
    b = buoyancy(model = model, temperature = T)
    
    d = solid_velocity_relaxation(model = model, temperature = T)
    
    _, psi_u, _ = fe.TestFunctions(solution.function_space())
        
    return dot(psi_u, u_t + grad(u)*u + b + d*u) \
        - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), sym(grad(u)))
    
    
def energy(model, solution):
    
    Pr = model.prandtl_number
    
    Ste = model.stefan_number
    
    _, u, T = fe.split(solution)
    
    phil = liquid_volume_fraction(model = model, temperature = T)
    
    rho_sl = model.density_solid_to_liquid_ratio
    
    c_sl = model.heat_capacity_solid_to_liquid_ratio
    
    C = phase_dependent_material_property(rho_sl*c_sl)(phil)
    
    k_sl = model.thermal_conductivity_solid_to_liquid_ratio
    
    k = phase_dependent_material_property(k_sl)(phil)
    
    _, CT_t, phil_t = time_discrete_terms(model = model)
    
    _, _, psi_T = fe.TestFunctions(solution.function_space())
    
    return psi_T*(CT_t + 1./Ste*phil_t + dot(u, grad(C*T))) \
        + 1./Pr*dot(grad(psi_T), k*grad(T))
        
    
def stabilization(model, solution):

    p, _, _ = fe.split(solution)
    
    psi_p, _, _ = fe.TestFunctions(solution.function_space())
    
    gamma = model.pressure_penalty_factor
    
    return gamma*psi_p*p
    
    
def variational_form_residual(model, solution):
    
    return sum(
            [r(model = model, solution = solution) 
            for r in (mass, momentum, energy, stabilization)])\
        *fe.dx(degree = model.quadrature_degree)

    
def plotvars(model, solution = None):
    
    if solution is None:
    
        solution = model.solution
        
    V = fe.FunctionSpace(
        solution.function_space().mesh(),
        fe.FiniteElement("P", model.mesh.ufl_cell(), 1))
    
    p, u, T = solution.split()
    
    phil = fe.interpolate(liquid_volume_fraction(
        model = model, temperature = T), V)
    
    return (p, u, T, phil), \
        ("p", "\\mathbf{u}", "T", "\\phi_l"), \
        ("p", "u", "T", "phil")
    
    
class Model(fempy.model.Model):
    
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
                "snes_converged_reason": True,
                "ksp_type": "preonly", 
                "pc_type": "lu", 
                "mat_type": "aij",
                "pc_factor_mat_solver_type": "mumps"},
            **kwargs)
            
    def solve_with_auto_smoothing(self):
        
        s0 = self.smoothing.__float__()
        
        def solve_with_over_regularization(self, startval):
        
            return fempy.continuation.solve_with_over_regularization(
                solve = self.solve,
                solution = self.solution,
                regularization_parameter = self.smoothing,
                startval = startval)
        
        def solve_with_bounded_regularization_sequence(self):
        
            return fempy.continuation.solve_with_bounded_regularization_sequence(
                solve = self.solve,
                solution = self.solution,
                backup_solution = self.backup_solution,
                regularization_parameter = self.smoothing,
                initial_regularization_sequence = self.smoothing_sequence)
                
        if self.smoothing_sequence is None:
        
            self.solution, smax = solve_with_over_regularization(self, startval = None)
            
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
        
        phil = liquid_volume_fraction(model = self, temperature = T)
        
        self.liquid_area = fe.assemble(phil*fe.dx)
        
        return self
        
    def write_outputs(self, *args, **kwargs):
        
        super().write_outputs(*args, plotvars = plotvars, **kwargs)
        