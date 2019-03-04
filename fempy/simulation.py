import fempy.output


class Simulation():

    def __init__(self, model):
    
        self.model = model
        
    def run(self,
            endtime,
            report = True,
            write_solution = False,
            plot = False,
            update_initial_values = True,
            quiet = False):
        
        self.model.output_directory_path.mkdir(
            parents = True, exist_ok = True)
        
        solution_filepath = self.model.\
            output_directory_path.joinpath("solution").with_suffix(".pvd")
        
        if write_solution:
        
            solution_file = fe.File(str(solution_filepath))
        
        if update_initial_values:
        
            self.model.update_initial_values()
            
        if report:
            
            fempy.output.report(
                self.model, write_header = True)
        
        if write_solution:
        
            self.write_solution(solution_file)
        
        if plot:
            
            fempy.output.plot(self.model)
            
        while self.model.time.__float__() < (
                endtime - self.model.time_tolerance):
            
            self.model.time.assign(self.model.time + self.model.timestep_size)
                
            self.model.solve()
            
            if report:
            
                fempy.output.report(
                    self.model, write_header = True)
                
            if write_solution:
        
                self.output.write_solution(
                    self.model, solution_file)
                
            if plot:
            
                fempy.output.plot(self.model)
            
            self.model.push_back_solutions()
            
            if not quiet:
            
                print("Solved at time t = {0}".format(
                    self.model.time.__float__()))
            