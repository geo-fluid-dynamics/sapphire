""" A simulation class for binary alloy melting and solidification in enthalpy form 

Equations are simplified from a more general model, now assuming that c_sl = 1 and T_0 = T_m.

f_l(h,S_l) was regularized by convoluting it with a Gaussian kernel.
"""
import firedrake as fe
import sapphire.simulation


def enthalpy(sim, temperature, porosity):

    T = temperature

    phi_l = porosity
    
    T_m = sim.pure_liquidus_temperature
    
    Ste = sim.stefan_number
    
    return T - T_m + 1./Ste*phi_l


def liquidus_temperature(sim, liquid_solute_concentration):
    
    T_m = sim.pure_liquidus_temperature
    
    S_l = liquid_solute_concentration
    
    return T_m*(1. - S_l)
    
    
def liquidus_enthalpy(sim, liquid_solute_concentration):
    
    Ste = sim.stefan_number
    
    S_l = liquid_solute_concentration
    
    T_L = liquidus_temperature(sim = sim, liquid_solute_concentration = S_l)
    
    T_m = sim.pure_liquidus_temperature
    
    return T_L - T_m + 1./Ste
    
    
def mushy_layer_porosity(sim, enthalpy, liquid_solute_concentration):
    
    h = enthalpy
    
    S_l = liquid_solute_concentration

    T_m = sim.pure_liquidus_temperature

    Ste = sim.stefan_number
    
    return Ste*(h + T_m*S_l)


erf = fe.erf

def erfc(x):

    return 1. - erf(x)
    
def regularized_porosity(sim, enthalpy, liquid_solute_concentration):
    
    h = enthalpy
    
    S_l = liquid_solute_concentration
    
    T_m = sim.pure_liquidus_temperature
    
    Ste = sim.stefan_number
    
    h_L = liquidus_enthalpy(sim = sim, liquid_solute_concentration = S_l)
    
    f_l_mush = mushy_layer_porosity(
        sim = sim, enthalpy = h, liquid_solute_concentration = S_l)
    
    exp, sqrt, pi = fe.exp, fe.sqrt, fe.pi
    
    sigma = sim.porosity_smoothing
    
    return \
        0.5*(1. 
             - exp(-(f_l_mush - 1.)**2/(2.*Ste**2*sigma**2))
               *sqrt(2./pi)*Ste*sigma 
             + erf((f_l_mush - 1.)/(sqrt(2.)*Ste*sigma)) 
             + f_l_mush*erfc((f_l_mush - 1.)/(sqrt(2.)*Ste*sigma)))


def temperature(sim, enthalpy, liquid_solute_concentration):
    
    h = enthalpy
    
    S_l = liquid_solute_concentration
    
    phi_l = regularized_porosity(
        sim = sim, enthalpy = h, liquid_solute_concentration = S_l)
    
    Ste = sim.stefan_number
    
    T_m = sim.pure_liquidus_temperature
    
    return (h - 1./Ste*phi_l) + T_m


def mushy_layer_liquid_solute_concentration(sim, enthalpy, porosity):
    
    h = enthalpy
    
    phi_l = porosity
    
    T_m = sim.pure_liquidus_temperature
    
    Ste = sim.stefan_number
    
    return (1./Ste*phi_l - h)/T_m
    

dot, grad = fe.dot, fe.grad
    
    
def time_discrete_terms(sim, solutions, timestep_size):

    h_t, _ = sapphire.simulation.time_discrete_terms(
        solutions = solutions, timestep_size = timestep_size)
    
    (h, S_l), (h_n, S_l_n) = solutions
    
    phi_l = regularized_porosity(
        sim = sim, enthalpy = h, liquid_solute_concentration = S_l)
    
    phi_l_n = regularized_porosity(
        sim = sim, enthalpy = h_n, liquid_solute_concentration = S_l_n)
    
    S = S_l*phi_l
    
    S_n = S_l_n*phi_l_n
    
    S_t = (S - S_n)/timestep_size
    
    return h_t, S_t
    
    
def variational_form_residual(sim, solution):
    
    h, S_l = fe.split(solution)
    
    T = temperature(
        sim = sim, enthalpy = h, liquid_solute_concentration = S_l)
    
    h_t, S_t = time_discrete_terms(
        sim = sim, 
        solutions = sim.solutions, 
        timestep_size = sim.timestep_size)
    
    psi_h, psi_S_l = fe.TestFunctions(sim.solution.function_space())
    
    phi_l = regularized_porosity(
        sim = sim, enthalpy = h, liquid_solute_concentration = S_l)
    
    energy = psi_h*h_t + dot(grad(psi_h), grad(T))
    
    Le = sim.lewis_number
    
    solute = psi_S_l*S_t + 1./Le*dot(grad(psi_S_l), phi_l*grad(S_l))
    
    dx = fe.dx(degree = sim.quadrature_degree)
    
    return (energy + solute)*dx
    
    
diff, div = fe.diff, fe.div
    
def strong_residual(sim, solution):
    
    h, S_l = solution
    
    t = sim.time
    
    phi_l = regularized_porosity(
        sim = sim, enthalpy = h, liquid_solute_concentration = S_l)
    
    T = temperature(
        sim = sim, enthalpy = h, liquid_solute_concentration = S_l)
    
    energy = diff(h, t) - div(grad(T))
    
    Le = sim.lewis_number
    
    S = S_l*phi_l
    
    solute = diff(S, t) - 1./Le*div(phi_l*grad(S_l))
    
    return energy, solute
    

def element(cell, degree):

    scalar_element = fe.FiniteElement("P", cell, degree)
    
    return fe.MixedElement(scalar_element, scalar_element)

    
def plotvars(sim, solution = None):
    
    if solution is None:
    
        solution = sim.solution
    
    h, S_l = solution.split()
    
    phi_l = sim.postprocessed_regularized_porosity
    
    T = sim.postprocessed_temperature
    
    S = sim.postprocessed_bulk_solute_concentration
    
    T_L = sim.postprocessed_liquidus_temperature
    
    h_L = sim.postprocessed_liquidus_enthalpy
    
    return (h, S_l, phi_l, T, S, T_L, h_L), \
        ("h", "S_l", "\\phi_l", "T", "S", "T_L(S_l)", "h_L(S_l)"), \
        ("h", "S_l", "phil", "T", "S", "T_L", "h_L")
     
    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args, 
            mesh, 
            stefan_number,
            lewis_number,
            pure_liquidus_temperature,
            porosity_smoothing,
            element_degree = 1, 
            snes_max_iterations = 100,
            snes_absolute_tolerance = 1.e-9,
            snes_step_tolerance = 1.e-9,
            snes_linesearch_damping = 1.,
            snes_linesearch_maxstep = 1.,
            **kwargs):
        
        self.stefan_number = fe.Constant(stefan_number)
        
        self.lewis_number = fe.Constant(lewis_number)
        
        self.pure_liquidus_temperature = fe.Constant(
            pure_liquidus_temperature)
        
        self.porosity_smoothing = fe.Constant(porosity_smoothing)
            
        self.initial_temperature = fe.Constant(1.)   # (T_i - T_e)/(T_i - T_e)
        
        self.snes_max_iterations = snes_max_iterations
        
        self.snes_absolute_tolerance = snes_absolute_tolerance
        
        self.snes_step_tolerance = snes_step_tolerance
        
        self.snes_linesearch_damping = snes_linesearch_damping
        
        self.snes_linesearch_maxstep = snes_linesearch_maxstep
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            variational_form_residual = variational_form_residual,
            **kwargs)
            
        self.postprocessed_regularized_porosity = \
            fe.Function(self.postprocessing_function_space)
        
        self.postprocessed_temperature = \
            fe.Function(self.postprocessing_function_space)
            
        self.postprocessed_bulk_solute_concentration = \
            fe.Function(self.postprocessing_function_space)
            
        self.postprocessed_liquidus_temperature = \
            fe.Function(self.postprocessing_function_space)
            
        self.postprocessed_liquidus_enthalpy = \
            fe.Function(self.postprocessing_function_space)
            
        self.postprocessed_functions = (
            self.postprocessed_regularized_porosity,
            self.postprocessed_temperature,
            self.postprocessed_bulk_solute_concentration,
            self.postprocessed_liquidus_temperature,
            self.postprocessed_liquidus_enthalpy)
            
    def solve(self, *args, **kwargs):
        
        return super().solve(*args,
            parameters = {
                "snes_type": "newtonls",
                "snes_max_it": self.snes_max_iterations,
                "snes_monitor": None,
                "snes_abstol": self.snes_absolute_tolerance,
                "snes_stol": self.snes_step_tolerance,
                "snes_rtol": 0.,
                "snes_linesearch_type": "l2",
                "snes_linesearch_maxstep": self.snes_linesearch_maxstep,
                "snes_linesearch_damping": self.snes_linesearch_damping,
                "ksp_type": "preonly", 
                "pc_type": "lu", 
                "mat_type": "aij",
                "pc_factor_mat_solver_type": "mumps"},
            **kwargs)
            
    def postprocess(self):
    
        h, S_l = self.solution.split()
        
        
        phi_l = fe.interpolate(
            regularized_porosity(
                sim = self,
                enthalpy = h,
                liquid_solute_concentration = S_l),
            self.postprocessing_function_space)
            
        self.postprocessed_regularized_porosity = \
            self.postprocessed_regularized_porosity.assign(phi_l)
        
        
        T = fe.interpolate(
            temperature(
                sim = self,
                enthalpy = h,
                liquid_solute_concentration = S_l),
            self.postprocessing_function_space)
        
        self.postprocessed_temperature = \
            self.postprocessed_temperature.assign(T)
        
        
        S = fe.interpolate(
            S_l*phi_l,
            self.postprocessing_function_space)
        
        self.postprocessed_bulk_solute_concentration = \
            self.postprocessed_bulk_solute_concentration.assign(S)
        
        
        T_L = fe.interpolate(
            liquidus_temperature(
                sim = self,
                liquid_solute_concentration = S_l),
            self.postprocessing_function_space)
            
        self.postprocessed_liquidus_temperature = \
            self.postprocessed_liquidus_temperature.assign(T_L)
        
        
        h_L = fe.interpolate(
            liquidus_enthalpy(
                sim = self,
                liquid_solute_concentration = S_l),
            self.postprocessing_function_space)
            
        self.postprocessed_liquidus_enthalpy = \
            self.postprocessed_liquidus_enthalpy.assign(h_L)
        
        
        self.total_solute = fe.assemble(S*fe.dx)
        
        
        return self
        
    def write_outputs(self, *args, **kwargs):
        
        super().write_outputs(*args, plotvars = plotvars, **kwargs)
        