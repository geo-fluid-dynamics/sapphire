""" A simulation class for melting and solidification in enthalpy form """
import firedrake as fe
import sapphire.simulation


diff, dot, div, grad, erf, sqrt = \
    fe.diff, fe.dot, fe.div, fe.grad, fe.erf, fe.sqrt

def liquid_volume_fraction(sim, temperature):
    
    T = temperature
    
    T_m = sim.liquidus_temperature
    
    sigma = sim.liquidus_smoothing_factor
    
    return 0.5*(1. + erf((T - T_m)/(sigma*sqrt(2))))

    
def time_discrete_terms(sim):
    
    T_t = sapphire.simulation.time_discrete_terms(
        solutions = sim.solutions, timestep_size = sim.timestep_size)
        
    phil_t = sapphire.time_discretization.bdf(
        [liquid_volume_fraction(sim = sim, temperature = T_n)
            for T_n in sim.solutions],
        timestep_size = sim.timestep_size)
    
    return T_t, phil_t
    
    
def weak_form_residual(sim, solution):
    
    T = solution
    
    Ste = sim.stefan_number
    
    T_t, phil_t = time_discrete_terms(sim = sim)
    
    v = fe.TestFunction(T.function_space())
    
    dx = fe.dx(degree = sim.quadrature_degree)
    
    return (v*(T_t + 1./Ste*phil_t) + dot(grad(v), grad(T)))*dx
    
    
def strong_residual(sim, solution):
    
    T = solution
    
    t = sim.time
    
    Ste = sim.stefan_number
    
    phil = liquid_volume_fraction(sim = sim, temperature = T)
    
    return diff(T, t) - div(grad(T)) + 1./Ste*diff(phil, t)
    

def element(cell, degree):

    return fe.FiniteElement("P", cell, degree)
    
    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args, 
            mesh, 
            element_degree = 1, 
            stefan_number = 1.,
            liquidus_temperature = 0.,
            liquidus_smoothing_factor = 0.01,
            solver_parameters = {"ksp_type": "cg"},
            **kwargs):
        
        self.stefan_number = fe.Constant(stefan_number)
        
        self.liquidus_temperature = fe.Constant(liquidus_temperature)
        
        self.liquidus_smoothing_factor = fe.Constant(
            liquidus_smoothing_factor)
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            weak_form_residual = weak_form_residual,
            solver_parameters = solver_parameters,
            **kwargs)
            