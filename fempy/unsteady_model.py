""" An abstract class on which to base finite element models
with auxiliary data for unsteady simulations.
"""
import firedrake as fe
import fempy.model
import fempy.time_discretization
import matplotlib.pyplot as plt
import csv


class Model(fempy.model.Model):
    """ An abstract class on which to base finite element models
        with auxiliary data for unsteady (i.e. time-dependent) simulations.
    """
    def __init__(self, quadrature_degree, spatial_order, temporal_order):
        
        self.time = fe.Constant(0.)
        
        self.timestep_size = fe.Constant(1.)
        
        self.time_tolerance = 1.e-8
        
        self.temporal_order = temporal_order
        
        super().__init__(
            quadrature_degree = quadrature_degree,
            spatial_order = spatial_order)
        
        self.solution_file = None
        
    def init_initial_values(self):
        
        self.initial_values = [fe.Function(self.function_space)
            for i in range(self.temporal_order)]
        
    def init_solution(self):
    
        super().init_solution()
        
        self.init_initial_values()
        
        self.solution.assign(self.initial_values[0])
        
        self.init_time_discrete_terms()
        
    def init_time_discrete_terms(self):
        
        solutions = [fe.split(self.solution)]
        
        for iv in self.initial_values:
        
            solutions.append(fe.split(iv))
        
        time_discrete_terms = [
            fempy.time_discretization.bdf(
                [solutions[n][i] for n in range(len(solutions))],
                order = self.temporal_order,
                timestep_size = self.timestep_size)
            for i in range(len(solutions[0]))]
            
        if len(time_discrete_terms) == 1:
        
            self.time_discrete_terms = time_discrete_terms[0]
            
        else:
        
            self.time_discrete_terms = time_discrete_terms
        
    def push_back_initial_values(self):
        
        for i in range(len(self.initial_values) - 1):
        
            self.initial_values[-i - 1].assign(
                self.initial_values[-i - 2])
            
        self.initial_values[0].assign(self.solution)
            
    def run(self, endtime, report = True, plot = False):
        
        self.output_directory_path.mkdir(
            parents = True, exist_ok = True)
            
        if report:
            
            self.report(write_header = True)
        
        if plot:
            
            self.plot()
            
        while self.time.__float__() < (endtime - self.time_tolerance):
            
            self.time.assign(self.time + self.timestep_size)
                
            self.solve()
            
            if report:
            
                self.report(write_header = False)
                
            if plot:
                
                self.plot()
                
            if self.solution_file is not None:
                
                self.solution_file.write(
                    *self.solution.split(), time = self.time.__float__())
            
            self.push_back_initial_values()
            
            if not self.quiet:
            
                print("Solved at time t = " + str(self.time.__float__()))
            
    def report(self, write_header = True):
        
        repvars = vars(self).copy()
        
        for key in repvars.keys():
            
            if type(repvars[key]) is type(fe.Constant(0.)):
            
                repvars[key] = repvars[key].__float__()
        
        with self.output_directory_path.joinpath(
                    "report").with_suffix(".csv").open("a+") as csv_file:
            
            writer = csv.DictWriter(csv_file, fieldnames = repvars.keys())
            
            if write_header:
                
                writer.writeheader()
            
            writer.writerow(repvars)
            
    def plot(self):
        
        self.output_directory_path.mkdir(
                parents = True, exist_ok = True)
                
        for i, f in enumerate(self.solution.split()):
            
            fe.plot(f)
            
            plt.axis("square")
            
            plt.title(r"$w_" + str(i) + "$, $ t = " 
                + str(self.time.__float__()) + "$")
            
            filepath = self.output_directory_path.joinpath(
                "w" + str(i) + "_t" + str(self.time.__float__()).replace(".", "p")
                ).with_suffix(".png")
            
            print("Writing plot to " + str(filepath))
            
            plt.savefig(str(filepath))
            
            plt.close()
        