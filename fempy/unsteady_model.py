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
        
    def init_solutions(self):
    
        super().init_solutions()
        
        for i in range(self.temporal_order):
        
            self.solutions.append(fe.Function(self.function_space))
            
        self.init_time_discrete_terms()
        
    def update_initial_values(self):
    
        initial_values = self.initial_values()
        
        for solution in self.solutions:
        
            solution.assign(initial_values)
        
    def initial_values(self):
        """ Redefine this to return a fe.Function(self.function_space) 
        with the initial values.
        """
        assert(False)
    
    def init_time_discrete_terms(self):
        
        solutions = self.solutions
        
        time_discrete_terms = [
            fempy.time_discretization.bdf(
                [fe.split(solutions[n])[i] for n in range(len(solutions))],
                order = self.temporal_order,
                timestep_size = self.timestep_size)
            for i in range(len(fe.split(solutions[0])))]
            
        if len(time_discrete_terms) == 1:
        
            self.time_discrete_terms = time_discrete_terms[0]
            
        else:
        
            self.time_discrete_terms = time_discrete_terms
        
    def push_back_solutions(self):
        
        for i in range(len(self.solutions[1:])):
        
            self.solutions[-(i + 1)].assign(
                self.solutions[-(i + 2)])
            
    def run(self,
            endtime,
            report = True,
            write_solution = False,
            plot = False,
            update_initial_values = True):
        
        self.output_directory_path.mkdir(
            parents = True, exist_ok = True)
            
        if update_initial_values:
        
            self.update_initial_values()
            
        if report:
            
            self.report(write_header = True)
        
        solution_filepath = self.output_directory_path.joinpath("solutions/")
        
        solution_file = fe.File(self.output_directory_path.joinpath("solution").with_suffix(".pvd"))
    
        if write_solution:
        
            self.write_solution(solution_file)
        
        if plot:
            
            self.plot()
            
        while self.time.__float__() < (endtime - self.time_tolerance):
            
            self.time.assign(self.time + self.timestep_size)
                
            self.solve()
            
            if report:
            
                self.report(write_header = False)
                
            if write_solution:
        
                self.write_solution(solution_file)
                
            if plot:
                
                self.plot()
                
            if self.solution_file is not None:
                
                self.solution_file.write(
                    *self.solution.split(), time = self.time.__float__())
            
            self.push_back_solutions()
            
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
        