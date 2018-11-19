""" An abstract class on which to base finite element models
with auxiliary data for unsteady (i.e. time-dependent) simulations.
"""
import firedrake as fe
import fem.model


class UnsteadyModel(fem.model.Model):
    """ An abstract class on which to base finite element models
        with auxiliary data for unsteady (i.e. time-dependent) simulations.
    """
    def __init__(self):
        
        self.time = fe.Constant(0.)
        
        self.timestep_size = fe.Constant(1.)
        
        super().__init__()
        
    def init_solution(self):
    
        super().init_solution()
        
        self.initial_values = fe.Function(self.function_space)
