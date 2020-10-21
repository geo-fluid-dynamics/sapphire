"""A simulation class using the enthalpy method.

Use this for melting and solidification.
"""
import firedrake as fe
import sapphire.simulation

    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args, 
            element_degree = 1, 
            stefan_number = 1.,
            liquidus_temperature = 0.,
            liquidus_smoothing_factor = 0.01,
            solver_parameters = {'ksp_type': 'cg'},
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
            fieldnames = 'T',
            **kwargs)
        
        self.time_discrete_terms['phi_l'] = sapphire.time_discretization.bdf(
            [self.liquid_volume_fraction(temperature = T_n) 
             for T_n in self.solutions],
            timestep_size = self.timestep_size)
    
    def liquid_volume_fraction(self, temperature):
    
        T = temperature
        
        T_m = self.liquidus_temperature
        
        sigma = self.liquidus_smoothing_factor
        
        erf, sqrt = fe.erf, fe.sqrt
        
        return 0.5*(1. + erf((T - T_m)/(sigma*sqrt(2))))
        
    def weak_form_residual(self):
        
        T = self.solution
        
        Ste = self.stefan_number
        
        T_t = self.time_discrete_terms['T']
        
        phil_t = self.time_discrete_terms['phi_l']
        
        v = fe.TestFunction(T.function_space())
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        dot, div, grad = fe.dot, fe.div, fe.grad
        
        return (v*(T_t + 1./Ste*phil_t) + dot(grad(v), grad(T)))*dx
