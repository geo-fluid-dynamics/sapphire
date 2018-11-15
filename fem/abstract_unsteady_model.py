""" **abstract_unsteady_model.py**
provides an abstract class on which to base finite element models
with auxiliary data for unsteady (i.e. time-dependent) simulations.
"""
import firedrake as fe
import fem.abstract_model


class AbstractUnsteadyModel(fem.abstract_model.AbstractModel):
    """ An abstract class on which to base finite element models
        with auxiliary data for unsteady (i.e. time-dependent) simulations.
    """
    def __init__(self):
        
        self.time = None
        
        self.ufl_time = fe.variable(0.)
        
        self.timestep_size = fe.Constant(1.)
        
        super().__init__()
        
    def init_solution(self):
    
        super().init_solution()
        
        self.initial_values = fe.Function(self.function_space)
        
    def set_initial_values(self, initial_values):
    
        if type(initial_values) is type(self.initial_values):
        
            self.initial_values.assign(initial_values)
            
        else:
            
            self.initial_values.assign(fe.interpolate(
                initial_values, self.function_space))
           