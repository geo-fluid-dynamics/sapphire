""" An abstract class on which to base finite element models
with auxiliary data for unsteady (i.e. time-dependent) simulations.
"""
import firedrake as fe
import fempy.model


TIME_EPSILON = 1.e-8

class Model(fempy.model.Model):
    """ An abstract class on which to base finite element models
        with auxiliary data for unsteady (i.e. time-dependent) simulations.
    """
    def __init__(self):
        
        self.time = fe.Constant(0.)
        
        self.timestep_size = fe.Constant(1.)
        
        super().__init__()
        
    def init_solution(self):
    
        super().init_solution()
        
        self.initial_values = [fe.Function(self.function_space),]
        
    def assign_initial_values(self, expression):
        
        # `fe.interpolate` will only work on one `ufl.Expr` at a time
        # So what do? Apparently we hadn't tested a time dependent system of PDE's_i
        # but rather just a time dependent scalar PDE.
        initial_values = fe.interpolate(
            expression, self.function_space)
                
        for iv in self.initial_values:
        
            iv.assign(initial_values)
        
    def run(self, endtime):
        
        while self.time.__float__() < (endtime - TIME_EPSILON):
            
            self.run_timestep()
            
            print("Solved time t = " + str(self.time.__float__()))
            