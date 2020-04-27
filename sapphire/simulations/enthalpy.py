"""A simulation class using the enthalpy method.

Use this for melting and solidification.
"""
import firedrake as fe
import sapphire.simulation


diff, dot, div, grad, erf, sqrt = \
    fe.diff, fe.dot, fe.div, fe.grad, fe.erf, fe.sqrt
    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args, 
            element_degree = 1, 
            stefan_number = 1.,
            liquidus_temperature = 0.,
            liquidus_smoothing_factor = 0.01,
            solver_parameters = {"ksp_type": "cg"},
            **kwargs):
        
        if "solution" not in kwargs:
            
            mesh = kwargs["mesh"]
            
            del kwargs["mesh"]
            
            element = fe.FiniteElement("P", mesh.ufl_cell(), element_degree)
            
            kwargs["solution"] = fe.Function(fe.FunctionSpace(mesh, element))
            
        self.stefan_number = fe.Constant(stefan_number)
        
        self.liquidus_temperature = fe.Constant(liquidus_temperature)
        
        self.liquidus_smoothing_factor = fe.Constant(
            liquidus_smoothing_factor)
        
        super().__init__(*args,
            solver_parameters = solver_parameters,
            **kwargs)            
    
    def liquid_volume_fraction(self, temperature):
    
        T = temperature
        
        T_m = self.liquidus_temperature
        
        sigma = self.liquidus_smoothing_factor
        
        return 0.5*(1. + erf((T - T_m)/(sigma*sqrt(2))))
    
    def time_discrete_terms(self):
        
        T_t = super().time_discrete_terms()
        
        phil_t = sapphire.time_discretization.bdf(
            [self.liquid_volume_fraction(temperature = T_n)
                for T_n in self.solutions],
            timestep_size = self.timestep_size)
        
        return T_t, phil_t
    
    def weak_form_residual(self):
        
        T = self.solution
        
        Ste = self.stefan_number
        
        T_t, phil_t = self.time_discrete_terms()
        
        v = fe.TestFunction(T.function_space())
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return (v*(T_t + 1./Ste*phil_t) + dot(grad(v), grad(T)))*dx


def strong_residual(sim, solution):
    
    T = solution
    
    t = sim.time
    
    Ste = sim.stefan_number
    
    phil = sim.liquid_volume_fraction(temperature = T)
    
    return diff(T, t) - div(grad(T)) + 1./Ste*diff(phil, t)
    