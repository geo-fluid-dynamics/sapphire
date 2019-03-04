""" Extends the Model class for time dependence """
import firedrake as fe
import fempy.model
import fempy.time_discretization


class UnsteadyModel(fempy.model.Model):
    """ A time dependent simulation based on a finite element model """
    def __init__(self, *args, time_stencil_size, **kwargs):
        
        self.time_stencil_size = time_stencil_size
        
        self.time = fe.Constant(0.)
        
        self.timestep_size = fe.Constant(1.)
        
        self.time_tolerance = 1.e-8
        
        super().__init__(*args, **kwargs)
        
    def init_solution(self):
        
        self.solutions = [fe.Function(self.function_space) 
            for i in range(self.time_stencil_size)]
        
        self.solution = self.solutions[0]
        
        self.init_time_discrete_terms()
    
    def init_time_discrete_terms(self):
        
        solutions = self.solutions
        
        time_discrete_terms = [
            fempy.time_discretization.bdf(
                [fe.split(solutions[n])[i] for n in range(len(solutions))],
                order = self.time_stencil_size - 1,
                timestep_size = self.timestep_size)
            for i in range(len(fe.split(solutions[0])))]
            
        if len(time_discrete_terms) == 1:
        
            self.time_discrete_terms = time_discrete_terms[0]
            
        else:
        
            self.time_discrete_terms = time_discrete_terms
        
    def initial_values(self):
        """ Redefine to return values on a fe.Function(self.function_space)"""
        assert(False)
        
    def update_initial_values(self):
    
        initial_values = self.initial_values()
        
        for solution in self.solutions:
        
            solution.assign(initial_values)
        
    def push_back_solutions(self):
        
        for i in range(len(self.solutions[1:])):
        
            self.solutions[-(i + 1)].assign(
                self.solutions[-(i + 2)])
