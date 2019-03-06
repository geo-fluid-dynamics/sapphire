""" A enthalpy-porosity model class for convection-coupled phase-change """
import firedrake as fe
import fempy.model
import fempy.continuation


tanh = fe.tanh

def liquid_volume_fraction(model, temperature):
    
    T = temperature
    
    T_L = model.liquidus_temperature
    
    s = model.smoothing
    
    return 0.5*(1. + tanh((T - T_L)/s))
    
    
def phase_dependent_material_property(solid_to_liquid_ratio):

    a_s = solid_to_liquid_ratio
    
    def a(phil):
    
        return a_s + (1. - a_s)*phil
    
    return a
    
    
def default_buoyancy(model, temperature):
    """ Boussinesq buoyancy """
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

def strong_residual(model, solution):
    
    Pr = model.prandtl_number
    
    Ste = model.stefan_number
    
    t = model.time
    
    p, u, T = solution
    
    b = buoyancy(model = model, temperature = T)
    
    d = solid_velocity_relaxation(model = model, temperature = T)
    
    phil = liquid_volume_fraction(model = model, temperature = T)
    
    cp = phase_dependent_material_property(
        model.heat_capacity_solid_to_liquid_ratio)(phil)
    
    k = phase_dependent_material_property(
        model.thermal_conductivity_solid_to_liquid_ratio)(phil)
        
    r_p = div(u)
    
    r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(sym(grad(u))) \
        + b + d*u
    
    r_T = diff(cp*T, t) + dot(u, grad(cp*T)) \
        - 1./Pr*div(k*grad(T)) + 1./Ste*diff(cp*phil, t)
    
    return r_p, r_u, r_T
    
    
def time_discrete_terms(model):
    
    _, u_t, _ = fempy.model.time_discrete_terms(
        solutions = model.solutions, timestep_size = model.timestep_size)
    
    temperature_solutions = []
    
    for solution in model.solutions:
    
        temperature_solutions.append(fe.split(solution)[2])
    
    def phil(T):
    
        return liquid_volume_fraction(model = model, temperature = T)
    
    cp = phase_dependent_material_property(
        model.heat_capacity_solid_to_liquid_ratio)
        
    cpT_t = fempy.time_discretization.bdf(
        [cp(phil(T))*T for T in temperature_solutions],
        timestep_size = model.timestep_size)
    
    cpphil_t = fempy.time_discretization.bdf(
        [cp(phil(T))*phil(T) for T in temperature_solutions],
        timestep_size = model.timestep_size)
    
    return u_t, cpT_t, cpphil_t
    

def mass(model, solution):
    
    _, u, _ = fe.split(solution)
    
    psi_p, _, _ = fe.TestFunctions(solution.function_space())
    
    div = fe.div
    
    mass = psi_p*div(u)
    
    return mass
    
    
def momentum(model, solution, buoyancy = default_buoyancy):
    
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
    
    cp = phase_dependent_material_property(
        model.heat_capacity_solid_to_liquid_ratio)(phil)
    
    k = phase_dependent_material_property(
        model.thermal_conductivity_solid_to_liquid_ratio)(phil)
    
    _, cpT_t, cpphil_t = time_discrete_terms(model = model)
    
    _, _, psi_T = fe.TestFunctions(solution.function_space())
    
    return psi_T*(cpT_t + dot(u, cp*grad(T)) + 1./Ste*cpphil_t) \
        + dot(grad(psi_T), k/Pr*grad(T))
        
    
def stabilization(model, solution):

    p, _, _ = fe.split(solution)
    
    psi_p, _, _ = fe.TestFunctions(solution.function_space())
    
    gamma = model.pressure_penalty_factor
    
    return gamma*psi_p*p
    
    
def variational_form_residual(model, solution):
    
    return sum(
            [r(model = model, solution = solution) 
            for r in (mass, momentum, energy, stabilization)])\
        *model.integration_measure

    
def plotvars(model, solution):
    
    V = fe.FunctionSpace(
        solution.function_space().mesh(),
        fe.FiniteElement("P", model.mesh.ufl_cell(), 1))
    
    p, u, T = solution.split()
    
    phil = fe.interpolate(liquid_volume_fraction(
        model = model, temperature = T), V)
    
    return (p, u, T, phil), \
        ("p", "\\mathbf{u}", "T", "\\phi_l"), \
        ("p", "u", "T", "phil")
        
        
def plot(model, solution = None):

    return fempy.output.plot(
        model = model,
        solution = solution,
        plotvars = lambda u: plotvars(model = model, solution = u))
        
        
def element(cell, degree):

    scalar = fe.FiniteElement("P", cell, degree)
    
    vector = fe.VectorElement("P", cell, degree)
    
    return fe.MixedElement(scalar, vector, scalar)
    
    
class Model(fempy.model.Model):
    
    def __init__(self, *args,
            mesh, element_degree, quadrature_degree = 8,
            **kwargs):
        
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
            
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            variational_form_residual = variational_form_residual,
            integration_measure = fe.dx(degree = quadrature_degree),
            **kwargs)
        
        self.backup_solution = fe.Function(self.solution)
        
        self.liquid_area = None
    
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
        
            self.solution, self.snes_iteration_count = super().solve()
           
        else:
        
            assert(self.smoothing.__float__()
                == self.smoothing_sequence[-1])
        
            for s in self.smoothing_sequence:
            
                self.smoothing.assign(s)
                
                self.solution, self.snes_iteration_count = super().solve()
                
                if not self.quiet:
                    
                    print("Solved with s = " + str(s))
    
        return self.solution, self.snes_iteration_count
        
    def postprocess(self):
        
        _, _, T = self.solution.split()
    
        phil = liquid_volume_fraction(self, T)
        
        self.liquid_area = fe.assemble(phil*self.integration_measure)
        
        return self
    