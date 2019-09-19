import csv
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')  # Only use back-end to prevent displaying image
import matplotlib.pyplot as plt
import firedrake as fe


def write_solution(sim, solution = None, time = None, file = None):
    
    if solution is None:
    
        solution = sim.solution
        
    if time is None:
    
        time = sim.time
        
    if file is None:
    
        time = sim.solution_file
        
        
    if time is None:
        
        file.write(*solution.split())
        
    else:
    
        if type(time) is type(0.):
        
            timefloat = time
            
        elif type(time) is fe.Constant:
        
            timefloat = time.__float__()
        
        file.write(*solution.split(), time = timefloat)
    

def default_plotvars(sim, solution = None):
    
    if solution is None:
    
        solution = sim.solution
        
    solution_functions = solution.split()
    
    labels = []
    
    for i in range(len(solution_functions)):
    
        labels.append("w{0}".format(i))
    
    filenames = labels
    
    return solution_functions, labels, filenames
    
    
def plot(
        sim,
        solution = None,
        time = None,
        outdir_path = None,
        plotvars = None):
    
    if solution is None:
    
        solution = sim.solution
        
    if time is None:
    
        time = sim.time.__float__()
        
    if outdir_path is None:
    
        outdir_path = sim.output_directory_path
    
    if plotvars is None:
    
        plotvars = default_plotvars
    
    outdir_path.mkdir(parents = True, exist_ok = True)
    
    for f, label, name in zip(*plotvars(sim = sim, solution = solution)):
        
        fe.plot(f)
        
        plt.axis("square")
        
        title = "${0}$".format(label)
        
        if time is not None:
        
            title += ", $t = {0}$".format(time)
        
        plt.title(title)
        
        filename = "{0}_t{1}".format(name, str(time).replace(".", "p"))
        
        filepath = outdir_path.joinpath(filename).with_suffix(".png")
            
        print("Writing plot to {0}".format(filepath))
        
        plt.savefig(str(filepath))
        
        plt.close()
        
        
class ObjectWithOrderedDict(object):
    """ Base class for maintaining an ordered dict of all attributes.
    See https://stackoverflow.com/questions/37591180/get-instance-variables-in-order-in-python
    """
    def __new__(Class, *args, **kwargs):
    
        instance = object.__new__(Class)
        
        instance.__odict__ = OrderedDict()
        
        return instance

    def __setattr__(self, key, value):
    
        if not key == "__odict__":
        
            self.__odict__[key] = value
            
        object.__setattr__(self, key, value)

    def keys(self):
    
        return self.__odict__.keys()

    def iteritems(self):
    
        return self.__odict__.iteritems()
        
        
def report(sim, write_header = True):
    
    ordered_dict = sim.__odict__.copy()
    
    for key in ordered_dict.keys():
        
        if type(ordered_dict[key]) is type(fe.Constant(0.)):
        
            ordered_dict[key] = ordered_dict[key].__float__()
    
    with sim.output_directory_path.joinpath(
                "report").with_suffix(".csv").open("a+") as csv_file:
        
        writer = csv.DictWriter(csv_file, fieldnames = ordered_dict.keys())
        
        if write_header:
            
            writer.writeheader()
        
        writer.writerow(ordered_dict)
        