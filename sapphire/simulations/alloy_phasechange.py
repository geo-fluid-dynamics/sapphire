""" A simulation class for binary alloy melting and solidification in enthalpy form """
import firedrake as fe
import sapphire.simulation

    
def liquid_volume_fraction(sim, enthalpy, solute_concentration):
    
    h = enthalpy
    
    S = solute_concentration
    
    T_m = sim.pure_liquidus_temperature
    
    m = -T_m
    
    Ste = sim.stefan_number
    
    h_L = m*S + 1./Ste
    
    c_sl = sim.heat_capacity_solid_to_liquid_ratio
    
    A = 1./Ste
    
    B = (1. - c_sl)*m*S - h
    
    C = c_sl*m*S
    
    sqrt = fe.sqrt
    
    return fe.conditional(
        h >= h_L, 1., (-B + sqrt(B**2. - 4.*A*C))/(2.*A))


def phase_dependent_material_property(solid_to_liquid_ratio):

    P_sl = solid_to_liquid_ratio
    
    def P(phil):
    
        return P_sl + (1. - P_sl)*phil
    
    return P
    

def temperature(sim, enthalpy, solute_concentration):
    
    h = enthalpy
    
    S = solute_concentration
    
    phi_l = liquid_volume_fraction(
        sim = sim, enthalpy = h, solute_concentration = S)
    
    Ste = sim.stefan_number
    
    c_sl = sim.heat_capacity_solid_to_liquid_ratio
    
    c = phase_dependent_material_property(c_sl)(phi_l)
    
    T_m = sim.pure_liquidus_temperature
    
    return (h - 1./Ste*phi_l)/c + T_m
    

dot, grad = fe.dot, fe.grad
    
def variational_form_residual(sim, solution):
    
    h, S = fe.split(solution)
    
    T = temperature(sim = sim, enthalpy = h, solute_concentration = S)
    
    h_t, S_t = sapphire.simulation.time_discrete_terms(
        solutions = sim.solutions, timestep_size = sim.timestep_size)
    
    psi_h, psi_S = fe.TestFunctions(sim.solution.function_space())
    
    phi_l = liquid_volume_fraction(
        sim = sim, enthalpy = h, solute_concentration = S)
        
    k_sl = sim.thermal_conductivity_solid_to_liquid_ratio
    
    k = phase_dependent_material_property(k_sl)(phi_l)
    
    energy = psi_h*h_t + dot(grad(psi_h), k*grad(T))
    
    Le = sim.lewis_number
    
    solute = psi_S*S_t + 1./Le*dot(grad(psi_S), phi_l*grad(S/phi_l))
    
    dx = fe.dx(degree = sim.quadrature_degree)
    
    return (energy + solute)*dx
    
    
diff, div = fe.diff, fe.div
    
def strong_residual(sim, solution):
    
    h, S = solution
    
    t = sim.time
    
    phi_l = liquid_volume_fraction(
        sim = sim, enthalpy = h, solute_concentration = S)
        
    k_sl = sim.thermal_conductivity_solid_to_liquid_ratio
    
    k = phase_dependent_material_property(k_sl)(phi_l)
    
    T = temperature(sim = sim, enthalpy = h, solute_concentration = S)
    
    energy = diff(h, t) - div(k*grad(T))
    
    Le = sim.lewis_number
    
    solute = diff(S, t) - 1./Le*div(phi_l*grad(S/phi_l))
    
    return energy, solute
    

def element(cell, degree):

    scalar_element = fe.FiniteElement("P", cell, degree)
    
    return fe.MixedElement(scalar_element, scalar_element)
    
    
def plotvars(sim, solution = None):
    
    if solution is None:
    
        solution = sim.solution
        
    V = fe.FunctionSpace(
        solution.function_space().mesh(),
        fe.FiniteElement("P", sim.mesh.ufl_cell(), 1))
    
    h, S = solution.split()
    
    phil = fe.interpolate(liquid_volume_fraction(
        sim = sim, enthalpy = h, solute_concentration = S), V)
    
    return (h, S, phil), \
        ("h", "S", "\\phi_l"), \
        ("h", "S", "phil")
        
    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args, mesh, element_degree = 1, **kwargs):
        
        self.stefan_number = fe.Constant(1.)
        
        self.lewis_number = fe.Constant(1.)
        
        self.pure_liquidus_temperature = fe.Constant(1.)
        
        self.heat_capacity_solid_to_liquid_ratio = fe.Constant(1.)
        
        self.thermal_conductivity_solid_to_liquid_ratio = fe.Constant(1.)
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            variational_form_residual = variational_form_residual,
            **kwargs)
            
    def solve(self, *args, **kwargs):
    
        return super().solve(*args,
            parameters = {
                "snes_type": "newtonls",
                "snes_max_it": 50,
                "snes_monitor": None,
                #"snes_linesearch_type": "l2",
                #"snes_linesearch_maxstep": 1.0,
                #"snes_linesearch_damping": 0.9,
                "snes_rtol": 0.,
                "ksp_type": "preonly", 
                "pc_type": "lu", 
                "mat_type": "aij",
                "pc_factor_mat_solver_type": "mumps"},
            **kwargs)
            
    def postprocess(self):
        
        h, S = self.solution.split()
        
        phil = liquid_volume_fraction(
            sim = self, enthalpy = h, solute_concentration = S)
        
        self.liquid_area = fe.assemble(phil*fe.dx)
        
        return self
        
    def write_outputs(self, *args, **kwargs):
        
        super().write_outputs(*args, plotvars = plotvars, **kwargs)
        