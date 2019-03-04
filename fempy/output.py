import matplotlib.pyplot as plt
import csv


def write_solution(model, file):
    
    if hasattr(model, "time"):
    
        file.write(*self.solution.split(), time = self.time)
        
    else:
    
        file.write(*self.solution.split())
    

def plot(model):
    
    outpath = model.output_directory_path
    
    outpath.mkdir(parents = True, exist_ok = True)
    
    if hasattr(model, "time"):
    
        time = model.time.__float__()
    
    for f, label, name in model.plotvars():
        
        fe.plot(f)
        
        plt.axis("square")
        
        title = label
        
        if hasattr(model, "time"):
        
            title += ", $ t = " + str(time) + "$"
        
        plt.title(title)
        
        model.output_directory_path.mkdir(
            parents = True, exist_ok = True)
    
        filename = name
        
        if hasattr(model, "time"):
        
            filename += "_t" + str(time).replace(".", "p")
        
        filepath = model.output_directory_path.joinpath(
            filename).with_suffix(".png")
            
        print("Writing plot to " + str(filepath))
        
        plt.savefig(str(filepath))
        
        plt.close()

        
def report(model, write_header = True):
    
    model.postprocess()
    
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
        