import xml.etree.ElementTree
import fempy.postprocessing.vtk
import fempy.test


datadir = fempy.test.datadir

vtk_filename = "freeze_water/solution_4.vtu"

def test__plot_mesh(datadir):
    
    data = fempy.postprocessing.vtk.read_vtk_data(
        vtk_filepath = str(datadir.join(vtk_filename)))
    
    axes = fempy.postprocessing.vtk.plot_mesh(vtk_data = data)
    
    outpath = datadir.join("mesh.png")
    
    print("Saving {0}".format(outpath))
    
    axes.get_figure().savefig(str(outpath))

    
def test__plot_scalar_field_contours(datadir):
    
    data = fempy.postprocessing.vtk.read_vtk_data(
        vtk_filepath = str(datadir.join(vtk_filename)))
    
    
    for filled in (False, True):
    
        axes, colorbar = fempy.postprocessing.vtk.plot_scalar_field_contours(
            vtk_data = data,
            scalar_solution_component = 2,
            filled = filled,
            levels = 8)
        
        colorbar.ax.set_title("$T$")
        
        name = "temperature_contours"
        
        if filled:
        
            name += "_filled"
        
        outpath = datadir.join(name + ".png")
        
        print("Saving {0}".format(outpath))
    
        axes.get_figure().savefig(str(outpath))
        
    
def test__plot_scalar_field(datadir):
    
    data = fempy.postprocessing.vtk.read_vtk_data(
        vtk_filepath = str(datadir.join(vtk_filename)))
        
    axes, colorbar = fempy.postprocessing.vtk.plot_scalar_field(
        vtk_data = data,
        scalar_solution_component = 2)
    
    colorbar.ax.set_title("$T$")
    
    outpath = datadir.join("temperature.png")
    
    print("Saving {0}".format(outpath))
    
    axes.get_figure().savefig(str(outpath))
    
    
def test__plot_vector_field(datadir):
    
    data = fempy.postprocessing.vtk.read_vtk_data(
        vtk_filepath = str(datadir.join(vtk_filename)))
        
    axes = fempy.postprocessing.vtk.plot_vector_field(
        vtk_data = data,
        vector_solution_component = 1,
        headwidth = 5)
    
    outpath = datadir.join("velocity.png")
    
    print("Saving {0}".format(outpath))
    
    axes.get_figure().savefig(str(outpath))
    
    
def test__plot_superposed_scalar_and_vector_fields(datadir):

    data = fempy.postprocessing.vtk.read_vtk_data(
        vtk_filepath = str(datadir.join(vtk_filename)))
        
    axes, colorbar = fempy.postprocessing.vtk.plot_scalar_field(
        vtk_data = data,
        scalar_solution_component = 2)
        
    colorbar.ax.set_title("$T$")
    
    axes = fempy.postprocessing.vtk.plot_vector_field(
        vtk_data = data,
        vector_solution_component = 1,
        axes = axes,
        headwidth = 5)
    
    outpath = datadir.join("temperature_and_velocity.png")
    
    print("Saving {0}".format(outpath))
    
    axes.get_figure().savefig(str(outpath))
    
    
vtk_dir = "freeze_water/"

pvd_filepath = vtk_dir + "solution.pvd"

def test__plot_unsteady_superposed_scalar_and_vector_fields(datadir):
    
    etree = xml.etree.ElementTree.parse(
        str(datadir.join(pvd_filepath))).getroot()
    
    for time, vtu_filename in [
            (element.attrib["timestep"], element.attrib["file"]) 
            for element in etree[0]]:
    
        data = fempy.postprocessing.vtk.read_vtk_data(
            vtk_filepath = str(datadir.join(vtk_dir + vtu_filename)))
            
        axes, colorbar = fempy.postprocessing.vtk.plot_scalar_field(
            vtk_data = data,
            scalar_solution_component = 2)
            
        colorbar.ax.set_title("$T$")
        
        axes = fempy.postprocessing.vtk.plot_vector_field(
            vtk_data = data,
            vector_solution_component = 1,
            axes = axes,
            headwidth = 5)
        
        axes.set_title("$t = {0}$".format(time))
        
        outpath = datadir.join(
            "temperature_and_velocity__t{0}.png".format(time))
        
        print("Saving {0}".format(outpath))
        
        axes.get_figure().savefig(str(outpath))
        