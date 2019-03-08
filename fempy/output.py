import firedrake as fe
import matplotlib.pyplot as plt
import csv


def write_solution(model, file):
    
    if hasattr(model, "time"):
    
        file.write(*model.solution.split(), time = model.time.__float__())
        
    else:
    
        file.write(*model.solution.split())
    

def default_plotvars(solution):
    
    subscripts, functions = enumerate(solution.split())
    
    labels = [r"$w_{0}$".format(i) for i in subscripts]
    
    filenames = ["w{0}".format(i) for i in subscripts]
    
    return functions, labels, filenames
    
    
def plot(model, solution = None, plotvars = default_plotvars):
    
    if solution is None:
    
        solution = model.solution
        
    outpath = model.output_directory_path
    
    outpath.mkdir(parents = True, exist_ok = True)
    
    time = model.time.__float__()
    
    for f, label, name in zip(*plotvars(solution)):
        
        fe.plot(f)
        
        plt.axis("square")
        
        title = "label, $ t = {0}$".format(time)
        
        plt.title(title)
        
        model.output_directory_path.mkdir(
            parents = True, exist_ok = True)
    
        filename = "{0}_t{1}".format(name, str(time).replace(".", "p"))
        
        filepath = model.output_directory_path.joinpath(
            filename).with_suffix(".png")
            
        print("Writing plot to " + str(filepath))
        
        plt.savefig(str(filepath))
        
        plt.close()

        
def report(model, postprocess = None, write_header = True):
    
    repvars = vars(model).copy()
    
    for key in repvars.keys():
        
        if type(repvars[key]) is type(fe.Constant(0.)):
        
            repvars[key] = repvars[key].__float__()
    
    if postprocess:
    
        for key, value in postprocess(model).items():
        
            repvars[key] = value
    
    with model.output_directory_path.joinpath(
                "report").with_suffix(".csv").open("a+") as csv_file:
        
        writer = csv.DictWriter(csv_file, fieldnames = repvars.keys())
        
        if write_header:
            
            writer.writeheader()
        
        writer.writerow(repvars)
        