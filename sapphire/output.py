"""Helper functions for simulation output"""
import typing
import datetime
import csv
import collections
import matplotlib
matplotlib.use('Agg')  # Only use back-end to prevent displaying image
import matplotlib.pyplot as plt
import firedrake as fe


def write_solution_to_vtk(
        sim, 
        solution = None, 
        dependent_functions = None, 
        time = None, 
        file = None):
        
    if solution is None:
    
        solution = sim.solution
    
    if time is None:
    
        time = sim.time
    
    if file is None:
    
        file = sim.solution_file
        
    functions_to_write = solution.split()
    
    if dependent_functions is None:
    
        if hasattr(sim, "postprocessed_functions"):
        
            dependent_functions = sim.postprocessed_functions
            
    if dependent_functions is not None:
    
        functions_to_write += dependent_functions
    
    
    print("Writing solution to {}".format(file.filename))
    
    
    if time is None:
        
        file.write(functions_to_write)
    
    else:
    
        if type(time) is type(0.):
        
            timefloat = time
            
        elif type(time) is fe.Constant:
        
            timefloat = time.__float__()
            
        file.write(*functions_to_write, time = timefloat)
        
        
def write_checkpoint(states, dirpath, filename, fieldname="solution"):
    """Write checkpoint for restarting and/or post-processing.
    
    A solution is stored for each state in states.
    
    Restarting requires states filling the time discretization stencil.
    
    If a checkpoint already exists for a state's time, then no new
    checkpoint is written for that state.
    """
    checkpointer = fe.DumbCheckpoint(
        basename=str(dirpath)+"/"+filename,
        mode=fe.FILE_UPDATE)
        
    stored_times, stored_indices = checkpointer.get_timesteps()
    
    for state in states:
        
        time = state["time"].__float__()
        
        if time in stored_times:
        
            continue
            
        else:
            
            solution = state["solution"]
            
            checkpointer.set_timestep(t=time, idx=state["index"])
            
            print("Writing checkpoint to {}".format(checkpointer.h5file.filename))
            
            checkpointer.store(solution, name=fieldname)


def read_checkpoint(states, dirpath, filename, fieldname="solution"):
    
    checkpointer = fe.DumbCheckpoint(
        basename=str(dirpath)+"/"+filename,
        mode=fe.FILE_READ)
        
    stored_times, stored_indices = checkpointer.get_timesteps()
    
    for state in states:
        
        time = state["time"].__float__()
        
        assert(time in stored_times)
        
        solution = state["solution"]
        
        checkpointer.set_timestep(t=time, idx=state["index"])
        
        print("Reading checkpoint from {}".format(checkpointer.h5file.filename))
        
        checkpointer.load(solution, name=fieldname)
        
    return states


def writeplots(
        fields: typing.List[typing.Union[fe.Function, fe.Mesh]],
        labels: typing.List[str],
        names: typing.List[str],
        plotfuns: typing.List[typing.Callable],
        time: typing.Union[float, None],
        time_index: int,
        outdir_path: str):
    """ Plot each field and write to files """
    outdir_path.mkdir(parents = True, exist_ok = True)
    
    for f, label, name, plot in zip(fields, labels, names, plotfuns):
        
        plot(f)
        
        title = "${}$".format(label)
        
        if time is not None:
        
            title += ", $t = {}$".format(time)
            
        plt.title(title)
        
        filename = "{}_it{}".format(name, time_index)
        
        filepath = outdir_path.joinpath(filename).with_suffix(".png")
        
        print("Writing plot to {}".format(filepath))
        
        plt.savefig(str(filepath))
        
        plt.close()
        
        
def report(sim, write_header = True):
    
    dict = sim.__dict__.copy()
    
    for key in dict.keys():
        
        if type(dict[key]) is type(fe.Constant(0.)):
        
            dict[key] = dict[key].__float__()
            
    dict["datetime"] = str(datetime.datetime.now())
    
    with sim.output_directory_path.joinpath(
            "report").with_suffix(".csv").open("a+") as csv_file:
        
        writer = csv.DictWriter(csv_file, fieldnames = dict.keys())
        
        if write_header:
            
            writer.writeheader()
            
        writer.writerow(dict)
        