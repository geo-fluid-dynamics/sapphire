""" An abstract class on which to base finite element models
with auxiliary data for unsteady (i.e. time-dependent) simulations.
"""
import firedrake as fe
import fempy.model
import abc
import matplotlib.pyplot as plt


class Model(fempy.model.Model):
    """ An abstract class on which to base finite element models
        with auxiliary data for unsteady (i.e. time-dependent) simulations.
    """
    def __init__(self):
        
        self.time = fe.Constant(0.)
        
        self.timestep_size = fe.Constant(1.)
        
        self.time_tolerance = 1.e-8
        
        super().__init__()
        
        self.solution_file = None
        
    def init_initial_values(self):
        
        self.initial_values = fe.Function(self.function_space)
        
    def init_solution(self):
    
        super().init_solution()
        
        self.init_initial_values()
        
        self.init_time_discrete_terms()
        
    def push_back_initial_values(self):
        
        if not((type(self.initial_values) == type((0,))) 
                or (type(self.initial_values) == type([0,]))):
        
            self.initial_values.assign(self.solution)

        else:
        
            for i in range(len(self.initial_values) - 1):
            
                self.initial_values[-i - 1].assign(
                    self.initial_values[-i - 2])
                
            self.initial_values[0].assign(self.solution)
        
    def run(self, endtime, plot = False):
        
        while self.time.__float__() < (endtime - self.time_tolerance):
            
            self.time.assign(self.time + self.timestep_size)
                
            self.solve()
            
            if plot:
                
                self.plot()
                
            if self.solution_file is not None:
                
                self.solution_file.write(
                    *self.solution.split(), time = self.time.__float__())
            
            self.push_back_initial_values()
            
            if not self.quiet:
            
                print("Solved at time t = " + str(self.time.__float__()))
            
    def plot(self):
        
        self.output_directory_path.mkdir(
                parents = True, exist_ok = True)
                
        for i, f in enumerate(self.solution.split()):
            
            fe.plot(f)
            
            plt.axis("square")
            
            plt.title(r"$w_" + str(i) + "$")
            
            filepath = self.output_directory_path.joinpath(
                "w" + str(i) + "_t" + str(self.time.__float__())
                ).with_suffix(".png")
            
            print("Writing plot to " + str(filepath))
            
            plt.savefig(str(filepath))
            
            plt.close()
        