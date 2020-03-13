"""A sim class for convection-coupled melting and solidification of binary alloys in enthalpy form"""
import firedrake as fe
import sapphire.simulation
import sapphire.continuation
import math
import numpy


def element(cell, degrees):
    """ Equal-order pressure, temperature, solute; increased order velocity"""
    pressure_element = fe.FiniteElement("P", cell, degrees[0])
    
    velocity_element = fe.VectorElement("P", cell, degrees[1])
    
    enthalpy_element = fe.FiniteElement("P", cell, degrees[2])
    
    solute_element = fe.FiniteElement("P", cell, degrees[3])
    
    return fe.MixedElement(
        pressure_element,
        velocity_element,
        enthalpy_element,
        solute_element)
    

def enthalpy(sim, temperature, porosity):

    T = temperature

    f_l = porosity
    
    T_m = sim.pure_liquidus_temperature
    
    Ste = sim.stefan_number
    
    return T - T_m + 1./Ste*f_l


def liquidus_temperature(sim, solute_concentration):
    
    T_m = sim.pure_liquidus_temperature
    
    S = solute_concentration
    
    return T_m*(1. - S)
    
    
def liquidus_enthalpy(sim, solute_concentration):
    
    Ste = sim.stefan_number
    
    S = solute_concentration
    
    T_L = liquidus_temperature(sim = sim, solute_concentration = S)
    
    T_m = sim.pure_liquidus_temperature
    
    return T_L - T_m + 1./Ste
    
    
def mushy_layer_porosity(sim, enthalpy, liquid_solute_concentration):
    
    h = enthalpy
    
    S_l = liquid_solute_concentration
    
    T_m = sim.pure_liquidus_temperature
    
    Ste = sim.stefan_number
    
    return Ste*(h + T_m*S_l)


def __porosity(sim, enthalpy, liquid_solute_concentration):
    """ The porosity function without regularization """
    h = enthalpy
    
    S_l = liquid_solute_concentration
    
    h_L = liquidus_enthalpy(sim = sim, solute_concentration = S_l)
    
    f_l_mush = mushy_layer_porosity(
        sim = sim, enthalpy = h, liquid_solute_concentration = S_l)
    
    return fe.conditional(h >= h_L, 1., f_l_mush)


erf = fe.erf

def erfc(x):

    return 1. - erf(x)
    
def regularized_porosity(sim, enthalpy, liquid_solute_concentration):
    
    h = enthalpy
    
    S_l = liquid_solute_concentration
    
    T_m = sim.pure_liquidus_temperature
    
    Ste = sim.stefan_number
    
    h_L = liquidus_enthalpy(sim = sim, solute_concentration = S_l)
    
    f_l_mush = mushy_layer_porosity(
        sim = sim, enthalpy = h, liquid_solute_concentration = S_l)
    
    exp, sqrt, pi = fe.exp, fe.sqrt, fe.pi
    
    sigma = sim.porosity_smoothing
    
    f_l = 0.5*(1. 
        - exp(-(f_l_mush - 1.)**2/(2.*Ste**2*sigma**2))*sqrt(2./pi)*Ste*sigma 
        + erf((f_l_mush - 1.)/(sqrt(2.)*Ste*sigma)) 
        + f_l_mush*erfc((f_l_mush - 1.)/(sqrt(2.)*Ste*sigma))) 
    
    if sim.enforce_minimum_porosity:
    
        f_l_min = sim.minimum_porosity
        
        f_l = fe.conditional(f_l < f_l_min, f_l_min, f_l)
    
    return f_l

def phase_dependent_material_property(solid_to_liquid_ratio):

    a_sl = solid_to_liquid_ratio
    
    def a(f_l):
    
        return a_sl + (1. - a_sl)*f_l
    
    return a
    

def temperature(sim, enthalpy, porosity):
    
    h = enthalpy
    
    f_l = porosity
    
    Ste = sim.stefan_number
    
    T_m = sim.pure_liquidus_temperature
    
    return (h - 1./Ste*f_l) + T_m


def mushy_layer_liquid_solute_concentration(sim, temperature):
    
    T = temperature
    
    T_m = sim.pure_liquidus_temperature
    
    S_l = 1. - T/T_m  # Mushy layer, T = T_L(S_l) = T_m*(1 - S_l)
    
    return S_l
    

def mushy_layer_bulk_solute_concentration(sim, temperature, porosity):
    
    S_l = mushy_layer_liquid_solute_concentration(
        sim = sim, temperature = temperature)
    
    f_l = porosity
    
    S = S_l*f_l
    
    return S
    
    
def buoyancy(sim, temperature, liquid_solute_concentration):
    
    T = temperature
    
    S_l = liquid_solute_concentration
    
    Ra_T = sim.thermal_rayleigh_number
    
    Ra_S = sim.solutal_rayleigh_number
    
    return Ra_T*T - Ra_S*S_l
    

def time_discrete_terms(sim):
    
    solutions = sim.solutions
    
    Delta_t = sim.timestep_size
    
    _, u_t, h_t, _ = sapphire.simulation.time_discrete_terms(
        solutions = solutions, timestep_size = Delta_t)
    
    (_, _, _, h, S_l), (_, _, _, h_n, S_l_n) = solutions
    
    phi_l = regularized_porosity(
        sim = sim, enthalpy = h, liquid_solute_concentration = S_l)
    
    phi_l_n = regularized_porosity(
        sim = sim, enthalpy = h_n, liquid_solute_concentration = S_l_n)
    
    S = S_l*phi_l
    
    S_n = S_l_n*phi_l_n
    
    S_t = (S - S_n)/Delta_t
    
    return u_t, h_t, S_t


dot, inner, grad, div, sym = fe.dot, fe.inner, fe.grad, fe.div, fe.sym

def mass(sim, solution):
    
    _, u, _, _ = fe.split(solution)
    
    psi_p, _, _, _ = fe.TestFunctions(solution.function_space())
    
    mass = psi_p*div(u)
    
    return mass
    
    
def momentum(sim, solution):
    """ Steady-state Darcy-Brinkman """
    p, u, h, S_l = fe.split(solution)
    
    Pr = sim.prandtl_number
    
    Da = sim.darcy_number
    
    phi_l = regularized_porosity(
        sim = sim, enthalpy = h, liquid_solute_concentration = S_l)
    
    T = temperature(
        sim = sim, enthalpy = h, porosity = phi_l)
    
    b = buoyancy(
        sim = sim, temperature = T, liquid_solute_concentration = S_l)
    
    gravdir = sim.gravity_direction
    
    ihat, jhat = sim.unit_vectors()
    
    ghat = gravdir[0]*ihat + gravdir[1]*jhat
    
    u_t, _, _ = time_discrete_terms(sim)
    
    _, psi_u, _, _ = fe.TestFunctions(solution.function_space())
    
    d = (1. - phi_l)**2/(Da*phi_l**2)
    
    return dot(psi_u, u_t + grad(u/phi_l)*u + Pr*(phi_l*b*ghat + d*u)) \
        - div(psi_u)*phi_l*p + Pr*inner(sym(grad(psi_u)), sym(grad(u)))
    
    
def energy(sim, solution):
    
    _, u, h, S_l = fe.split(solution)
    
    phi_l = regularized_porosity(
        sim = sim, enthalpy = h, liquid_solute_concentration = S_l)
    
    T = temperature(sim = sim, enthalpy = h, porosity = phi_l)
    
    k_sl = sim.thermal_conductivity_solid_to_liquid_ratio
    
    k = phase_dependent_material_property(k_sl)(phi_l)
    
    _, h_t, _ = time_discrete_terms(sim = sim)
    
    _, _, psi_h, _ = fe.TestFunctions(solution.function_space())
    
    return psi_h*(h_t + dot(u, grad(T))) + dot(grad(psi_h), k*grad(T))
    
    
def solute(sim, solution):
    
    _, u, h, S_l = fe.split(solution)
    
    Le = sim.lewis_number
    
    phi_l = regularized_porosity(
        sim = sim, enthalpy = h, liquid_solute_concentration = S_l)
    
    _, _, S_t = time_discrete_terms(sim = sim)
    
    _, _, _, psi_S = fe.TestFunctions(solution.function_space())
    
    return psi_S*(S_t + dot(u, grad(S_l))) \
        + 1./Le*dot(grad(psi_S), phi_l*grad(S_l))
    
    
def stabilization(sim, solution):

    p, _, _, _ = fe.split(solution)
    
    psi_p, _, _, _ = fe.TestFunctions(solution.function_space())
    
    gamma = sim.pressure_penalty_factor
    
    return gamma*psi_p*p
    
    
def variational_form_residual(sim, solution):
    
    return sum(
            [r(sim = sim, solution = solution) 
            for r in (mass, momentum, energy, solute, stabilization)])\
        *fe.dx(degree = sim.quadrature_degree)

    
def plotvars(sim, solution = None):
    
    if solution is None:
    
        solution = sim.solution
    
    p, u, h, S_l = solution.split()
    
    phi_l = sim.postprocessed_porosity
    
    T = sim.postprocessed_temperature
    
    S = sim.postprocessed_bulk_solute_concentration
    
    return (p, u, h, S_l, S, phi_l, T), \
        ("p", "\\mathbf{u}", "h", "S_l", "S", "\\phi_l", "T"), \
        ("p", "u", "h", "Sl", "S", "phil", "T")
    
default_solver_parameters = {
    "snes_type": "newtonls",
    "snes_max_it": 24,
    "snes_monitor": None,
    "snes_abstol": 1.e-9,
    "snes_stol": 1.e-9,
    "snes_rtol": 0.,
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxstep": 1.,
    "snes_linesearch_damping": 1.,
    "ksp_type": "preonly", 
    "pc_type": "lu", 
    "mat_type": "aij",
    "pc_factor_mat_solver_type": "mumps"}
    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args,
            mesh, 
            darcy_number,
            lewis_number,
            prandtl_number,
            thermal_rayleigh_number,
            solutal_rayleigh_number,
            stefan_number,
            pure_liquidus_temperature,
            thermal_conductivity_solid_to_liquid_ratio,
            porosity_smoothing,
            gravity_direction = (0., -1.),
            pressure_penalty_factor = 1.e-7,
            minimum_porosity = 0.,
            enforce_minimum_porosity = False,
            element_degrees = (1, 1, 1, 1), 
            solver_parameters = default_solver_parameters,
            adaptive_timestep_minimum = 1.e-6,
            **kwargs):
        
        self.darcy_number = fe.Constant(darcy_number)
        
        self.lewis_number = fe.Constant(lewis_number)
        
        self.prandtl_number = fe.Constant(prandtl_number)
        
        self.thermal_rayleigh_number = fe.Constant(thermal_rayleigh_number)
        
        self.solutal_rayleigh_number = fe.Constant(solutal_rayleigh_number)
        
        self.stefan_number = fe.Constant(stefan_number)
        
        self.pure_liquidus_temperature = fe.Constant(pure_liquidus_temperature)
        
        self.thermal_conductivity_solid_to_liquid_ratio = fe.Constant(
            thermal_conductivity_solid_to_liquid_ratio)
        
        self.max_temperature = fe.Constant(1.)   # (T_i - T_e)/(T_i - T_e)
        
        gravdir = gravity_direction
        
        gravdir_mag = math.sqrt(gravdir[0]**2 + gravdir[1]**2)
        
        self.gravity_direction = (
            gravdir[0]/gravdir_mag, gravdir[1]/gravdir_mag)  # Normalize to a unit vector
        
        
        self.pressure_penalty_factor = fe.Constant(pressure_penalty_factor)
        
        
        self.enforce_minimum_porosity = enforce_minimum_porosity
        
        self.minimum_porosity = fe.Constant(minimum_porosity)
        
        self.porosity_smoothing = fe.Constant(porosity_smoothing)
        
        
        self.adaptive_timestep_minimum = fe.Constant(adaptive_timestep_minimum)
        
        
        if "variational_form_residual" not in kwargs:
        
            kwargs["variational_form_residual"] = variational_form_residual
        
        if "time_stencil_size" not in kwargs:
        
            kwargs["time_stencil_size"] = 2  # Backward Euler
            
        self.element_degrees = element_degrees
            
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degrees = element_degrees),
            solution_name = "p_u_h_Sl",
            solver_parameters = solver_parameters,
            **kwargs)
        
        self.postprocessed_bulk_solute_concentration = \
            fe.Function(
                self.postprocessing_function_space,
                name = "S")
            
        self.postprocessed_porosity = \
            fe.Function(
                self.postprocessing_function_space,
                name = "phi_l")
        
        self.postprocessed_temperature = \
            fe.Function(
                self.postprocessing_function_space,
                name = "T")
        
        self.postprocessed_liquidus_enthalpy = \
            fe.Function(
                self.postprocessing_function_space,
                name = "h_L")
        
        self.postprocessed_functions = (
            self.postprocessed_bulk_solute_concentration,
            self.postprocessed_porosity,
            self.postprocessed_temperature,
            self.postprocessed_liquidus_enthalpy)
        
    def solve_with_adaptive_timestep(self, minimum):
        """ Disregard the originally intended timestep size
            and find a smaller size which is solvable. """
        
        while self.timestep_size.__float__() >= minimum:
        
            try:
                
                Delta_t = self.timestep_size.__float__()
                
                print("Attempting to solve with timestep size = {}".format(
                    Delta_t))
                
                self.solution = self.solve()
                
                return self.solution, self.timestep_size
            
            except fe.exceptions.ConvergenceError as exception:
                
                print("Failed to solve with timestep size = {}".format(
                    Delta_t))
                
                self.solution = self.solution.assign(self.solutions[1])
                
                self.timestep_size = self.timestep_size.assign(Delta_t/2.)
                
    def run(self,
            endtime,
            solve_with_adaptive_timestep = True,
            write_initial_outputs = True,
            validate_state = True):
        
        if write_initial_outputs:
        
            self.write_outputs(write_headers = True)
        
        if solve_with_adaptive_timestep:
        
            Delta_t_0 = self.timestep_size.__float__()
        
        while self.time.__float__() < (
                endtime - sapphire.simulation.time_tolerance):
            
            if solve_with_adaptive_timestep:
            
                t = self.time.__float__()
                
                Delta_t = self.timestep_size.__float__()
                
                latest_nexttime = t + Delta_t
                
                if latest_nexttime > endtime:
                
                    self.timestep_size = self.timestep_size.assign(endtime - t)
            
                self.solution, self.timestep_size = self.solve_with_adaptive_timestep(
                    minimum = self.adaptive_timestep_minimum.__float__())
            
            else:            
                
                self.solution = self.solve()
                
            self.time = self.time.assign(self.time + self.timestep_size)
            
            print("Solved at time t = {0}".format(self.time.__float__()))
            
            self.write_outputs(write_headers = False)
            
            if validate_state:
            
                self.validate_state()
            
            self.solutions = self.push_back_solutions()
            
            if solve_with_adaptive_timestep:
                
                next_Delta_t = 2.*self.timestep_size.__float__()
                
                if next_Delta_t > Delta_t_0:
                    
                    next_Delta_t = Delta_t_0
                
                self.timestep_size = self.timestep_size.assign(next_Delta_t)
            
        return self.solutions, self.time
        
    def postprocess(self):
    
        _, u, h, S_l = self.solution.split()
        
        
        phi_l = fe.interpolate(
            regularized_porosity(
                sim = self,
                enthalpy = h,
                liquid_solute_concentration = S_l),
            self.postprocessing_function_space)
            
        self.postprocessed_porosity = \
            self.postprocessed_porosity.assign(phi_l)
        
        
        self.minimum_porosity = numpy.min(phi_l.vector().array())
        
        
        S = fe.interpolate(
            S_l*phi_l,
            self.postprocessing_function_space)
        
        self.postprocessed_bulk_solute_concentration = \
            self.postprocessed_bulk_solute_concentration.assign(S)
            
        
        T = fe.interpolate(
            temperature(
                sim = self,
                enthalpy = h,
                porosity = phi_l),
            self.postprocessing_function_space)
        
        self.postprocessed_temperature = \
            self.postprocessed_temperature.assign(T)
        
        
        h_L = fe.interpolate(
            liquidus_enthalpy(sim = self, solute_concentration = S_l),
            self.postprocessing_function_space)
        
        self.postprocessed_liquidus_enthalpy = \
            self.postprocessed_liquidus_enthalpy.assign(h_L)
            
        
        self.liquid_area = fe.assemble(phi_l*fe.dx)
        
        self.total_solute = fe.assemble(S*fe.dx)
        
        self.max_speed = u.vector().max()
        
        return self
    
    def write_outputs(self, *args, **kwargs):
        
        super().write_outputs(*args, plotvars = plotvars, **kwargs)
        
    def validate_state(self):
        
        if self.minimum_porosity < 0.:
        
            raise ValueError(
                "The minimum porosity is {}.".format(self.minimum_porosity) +
                " The porosity must be everywhere greater than zero.")
        