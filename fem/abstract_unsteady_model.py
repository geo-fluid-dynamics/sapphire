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
    def __init__(self, *args, **kwargs):
        
        self.timestep_size = fe.Constant(0.)
        
        super().__init__(*args, **kwargs)
        
    def init_solution(self):
    
        super().init_solution()
        
        self.initial_values = fe.Function(self.solution.function_space())
        
    def set_initial_values(self, initial_values):
    
        if type(initial_values) is type(self.initial_values):
        
            self.initial_values.assign(initial_values)
            
        else:
            
            self.initial_values.assign(fe.interpolate(
                initial_values, self.solution.function_space()))
            