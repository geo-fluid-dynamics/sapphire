""" An abstract class on which to base finite element models
with auxiliary data for unsteady (i.e. time-dependent) simulations.
"""
import firedrake as fe
import fempy.model
import abc


class Model(fempy.model.Model):
    """ An abstract class on which to base finite element models
        with auxiliary data for unsteady (i.e. time-dependent) simulations.
    """
    def __init__(self):
        
        self.time = fe.Constant(0.)
        
        self.timestep_size = fe.Constant(1.)
        
        self.time_tolerance = 1.e-8
        
        super().__init__()
        
    @abc.abstractmethod
    def init_initial_values(self):
        """ Redefine this to set `self.initial_values` to a `fe.Function`. """
        
    def init_solution(self):
    
        super().init_solution()
        
        self.init_initial_values()
        
        self.init_time_discrete_terms()
        
        u0 = self.initial_values
        
        if not ((type(u0) == type((0,))) or (type(u0) == type([0,]))):
        
            u0 = (u0,)
        
        self.solution.assign(u0[0])
        
    def push_back_initial_values(self):
        
        if not((type(self.initial_values) == type((0,))) 
                or (type(self.initial_values) == type([0,]))):
        
            self.initial_values.assign(self.solution)

        else:
        
            for i in range(len(self.initial_values) - 1):
            
                self.initial_values[-i - 1].assign(
                    self.initial_values[-i - 2])
                
            self.initial_values[0].assign(self.solution)
        
    def run(self, endtime):
        
        while self.time.__float__() < (endtime - self.time_tolerance):
            
            self.time.assign(self.time + self.timestep_size)
                
            self.solve()
            
            self.push_back_initial_values()
            
            if not self.quiet:
            
                print("Solved at time t = " + str(self.time.__float__()))
            