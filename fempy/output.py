import firedrake as fe
import matplotlib.pyplot as plt
import csv


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
    
        file.write(*solution.split(), time = time)
    

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
        
        title = label
        
        if time is not None:
        
            title += ", $ t = {0}$".format(time)
        
        plt.title(title)
        
        filename = "{0}_t{1}".format(name, str(time).replace(".", "p"))
        
        filepath = outdir_path.joinpath(filename).with_suffix(".png")
            
        print("Writing plot to {0}".format(filepath))
        
        plt.savefig(str(filepath))
        
        plt.close()
        
        
def report(model, write_header = True):
    
    repvars = vars(model).copy()
    
    for key in repvars.keys():
        
        if type(repvars[key]) is type(fe.Constant(0.)):
        
            repvars[key] = repvars[key].__float__()
    
    with model.output_directory_path.joinpath(
                "report").with_suffix(".csv").open("a+") as csv_file:
        
        writer = csv.DictWriter(csv_file, fieldnames = repvars.keys())
        
        if write_header:
            
            writer.writeheader()
        
        writer.writerow(repvars)
        