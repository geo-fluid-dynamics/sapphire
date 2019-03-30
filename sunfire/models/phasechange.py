""" An enthalpy model class for melting and solidification """
import firedrake as fe
import sunfire.model


diff, dot, div, grad, tanh = fe.diff, fe.dot, fe.div, fe.grad, fe.tanh


def liquid_volume_fraction(model, temperature):
    
    T = temperature
    
    T_L = model.liquidus_temperature
    
    s = model.smoothing
    
    return 0.5*(1. + tanh((T - T_L)/s))

    
def time_discrete_terms(model):
    
    T_t = sunfire.model.time_discrete_terms(
        solutions = model.solutions, timestep_size = model.timestep_size)
        
    phil_t = sunfire.time_discretization.bdf(
        [liquid_volume_fraction(model = model, temperature = T_n)
            for T_n in model.solutions],
        timestep_size = model.timestep_size)
    
    return T_t, phil_t
    
    
def variational_form_residual(model, solution):
    
    T = solution
    
    Ste = model.stefan_number
    
    T_t, phil_t = time_discrete_terms(model = model)
    
    v = fe.TestFunction(T.function_space())
    
    dx = fe.dx(degree = model.quadrature_degree)
    
    return (v*(T_t + 1./Ste*phil_t) + dot(grad(v), grad(T)))*dx
    
    
def strong_residual(model, solution):
    
    T = solution
    
    t = model.time
    
    Ste = model.stefan_number
    
    phil = liquid_volume_fraction(model = model, temperature = T)
    
    return diff(T, t) - div(grad(T)) + 1./Ste*diff(phil, t)
    

def element(cell, degree):

    return fe.FiniteElement("P", cell, degree)
    
    
class Model(sunfire.model.Model):
    
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
            