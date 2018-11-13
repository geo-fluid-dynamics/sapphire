""" **abstract_unsteady_model.py**
provides an abstract class on which to base finite element models
with auxiliary data for unsteady (i.e. time-dependent) simulations.
"""
import firedrake as fe
import fempy.abstract_model


class AbstractUnsteadyModel(fempy.abstract_model):
    """ An abstract class on which to base finite element models
        with auxiliary data for unsteady (i.e. time-dependent) simulations.
    """
    def __init__(self, *args, **kwargs):
        
        self.time = fenics.Constant(0.)
        
        self.old_time = fenics.Constant(0.)
        
        super().__init__()
        
        
    def init_solution(self):
    
        super().init_solution()
        
        self.old_solution = fenics.Function(self.solution)
        