""" An abstract class on which to base finite element models
with auxiliary data for unsteady (i.e. time-dependent) simulations.
"""
import firedrake as fe
import fempy.model
import abc


TIME_EPSILON = 1.e-8

class Model(fempy.model.Model):
    """ An abstract class on which to base finite element models
        with auxiliary data for unsteady (i.e. time-dependent) simulations.
    """
    def __init__(self):
        
        self.time = fe.Constant(0.)
        
        self.timestep_size = fe.Constant(1.)
        
        super().__init__()
        
    @abc.abstractmethod
    def init_initial_values(self):
        """ Redefine this to set `self.initial_values` to a `fe.Function`. """
        
    def init_solution(self):
    
        super().init_solution()
        
        self.init_initial_values()
        
        self.init_time_discrete_terms()
        
        self.solution.assign(self.initial_values[0])
        
    def push_back_initial_values(self):
    
        for i in range(len(self.initial_values) - 1):
        
            self.initial_values[-i - 1].assign(
                self.initial_values[-i - 2])
                
        self.initial_values[0].assign(self.solution)
        
    def run(self, endtime):
        
        between_timesteps = False
        
        while self.time.__float__() < (endtime - TIME_EPSILON):
            
            if between_timesteps:
            
                self.time.assign(self.time + self.timestep_size)
                
                self.push_back_initial_values()
                
            self.solve()
            
            between_timesteps = True
            