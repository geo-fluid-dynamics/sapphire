import csv
from collections import OrderedDict
import matplotlib.pyplot as plt
import firedrake as fe


def write_solution(model, solution = None, time = None, file = None):
    
    if solution is None:
    
        solution = model.solution
        
    if time is None:
    
        time = model.time
        
    if file is None:
    
        time = model.solution_file
        
        
    if time is None:
        
        file.write(*solution.split())
        
    else:
    
        if type(time) is type(0.):
        
            timefloat = time
            
        elif type(time) is fe.Constant:
        
            timefloat = time.__float__()
        
        file.write(*solution.split(), time = timefloat)
    

def default_plotvars(model, solution = None):
    
    if solution is None:
    
        solution = model.solution
        
    subscripts, functions = enumerate(solution.split())
    
    labels = [r"$w_{0}$".format(i) for i in subscripts]
    
    filenames = ["w{0}".format(i) for i in subscripts]
    
    return functions, labels, filenames
    
    
def plot(
        model,
        solution = None,
        time = None,
        outdir_path = None,
        plotvars = None):
    
    if solution is None:
    
        solution = model.solution
        
    if time is None:
    
        time = model.time.__float__()
        
    if outdir_path is None:
    
        outdir_path = model.output_directory_path
    
    if plotvars is None:
    
        plotvars = default_plotvars
    
    outdir_path.mkdir(parents = True, exist_ok = True)
    
    for f, label, name in zip(*plotvars(model = model, solution = solution)):
        
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
        
        
def report(model, write_header = True):
    
    ordered_dict = model.__odict__.copy()
    
    for key in ordered_dict.keys():
        
        if type(ordered_dict[key]) is type(fe.Constant(0.)):
        
            ordered_dict[key] = ordered_dict[key].__float__()
    
    with model.output_directory_path.joinpath(
                "report").with_suffix(".csv").open("a+") as csv_file:
        
        writer = csv.DictWriter(csv_file, fieldnames = ordered_dict.keys())
        
        if write_header:
            
            writer.writeheader()
        
        writer.writerow(ordered_dict)
        