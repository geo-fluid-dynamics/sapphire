""" A simulation class for melting and solidification in enthalpy form """
import firedrake as fe
import sapphire.simulation


diff, dot, div, grad, tanh = fe.diff, fe.dot, fe.div, fe.grad, fe.tanh


def liquid_volume_fraction(sim, temperature):
    
    T = temperature
    
    T_L = sim.liquidus_temperature
    
    s = sim.smoothing
    
    return 0.5*(1. + tanh((T - T_L)/s))

    
def time_discrete_terms(sim):
    
    T_t = sapphire.simulation.time_discrete_terms(
        solutions = sim.solutions, timestep_size = sim.timestep_size)
        
    phil_t = sapphire.time_discretization.bdf(
        [liquid_volume_fraction(sim = sim, temperature = T_n)
            for T_n in sim.solutions],
        timestep_size = sim.timestep_size)
    
    return T_t, phil_t
    
    
def variational_form_residual(sim, solution):
    
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
    
    def __init__(self, *args, mesh, element_degree, **kwargs):
        
        self.stefan_number = fe.Constant(1.)
        
        self.liquidus_temperature = fe.Constant(0.)
        
        self.smoothing = fe.Constant(1./32.)
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            variational_form_residual = variational_form_residual,
            **kwargs)
            
    def solve(self, *args, **kwargs):
    
        return super().solve(*args, parameters = {"ksp_type": "cg"}, **kwargs)
            