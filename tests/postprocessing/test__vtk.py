from distutils import dir_util
from pytest import fixture
import os
import fempy.postprocessing.vtk


@fixture
def datadir(tmpdir, request):
    """    
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    
    Copied from https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    
    Dear future developers:
    - Should this be done with pathlib now?
    """
    filename = request.module.__file__
    
    testdir, _ = os.path.splitext(filename)

    if os.path.isdir(testdir):
    
        dir_util.copy_tree(testdir, str(tmpdir))

    return tmpdir
    

def test__plot_mesh(datadir):
    
    data = fempy.postprocessing.vtk.read_vtk_data(
        vtk_filepath = str(datadir.join("water_freezing_endtime.vtu")))
    
    axes = fempy.postprocessing.vtk.plot_mesh(vtk_data = data)
    
    outpath = datadir.join("mesh.png")
    
    print("Saving {0}".format(outpath))
    
    axes.get_figure().savefig(str(outpath))

    
def test__plot_scalar_field_contours(datadir):
    
    data = fempy.postprocessing.vtk.read_vtk_data(
        vtk_filepath = str(datadir.join("water_freezing_endtime.vtu")))
    
    
    for filled in (False, True):
    
        axes, colorbar = fempy.postprocessing.vtk.plot_field_contours(
            vtk_data = data,
            scalar_solution_component = 2,
            contours = 8,
            filled = filled)
        
        colorbar.ax.set_title("$T$")
        
        name = "temperature_contours"
        
        if filled:
        
            name += "_filled"
        
        outpath = datadir.join(name + ".png")
        
        print("Saving {0}".format(outpath))
    
        axes.get_figure().savefig(str(outpath))
        
    
def test__plot_scalar_field(datadir):
    
    data = fempy.postprocessing.vtk.read_vtk_data(
        vtk_filepath = str(datadir.join("water_freezing_endtime.vtu")))
        
    axes, colorbar = fempy.postprocessing.vtk.plot_scalar_field(
        vtk_data = data,
        scalar_solution_component = 2)
    
    colorbar.ax.set_title("$T$")
    
    outpath = datadir.join("temperature.png")
    
    print("Saving {0}".format(outpath))
    
    axes.get_figure().savefig(str(outpath))
    
    
def test__plot_vector_field(datadir):
    
    data = fempy.postprocessing.vtk.read_vtk_data(
        vtk_filepath = str(datadir.join("water_freezing_endtime.vtu")))
        
    axes = fempy.postprocessing.vtk.plot_vector_field(
        vtk_data = data,
        vector_solution_component = 1,
        headwidth = 5)
    
    outpath = datadir.join("velocity.png")
    
    print("Saving {0}".format(outpath))
    
    axes.get_figure().savefig(str(outpath))
    
    
def test__plot_superposed_scalar_and_vector_fields(datadir):

    data = fempy.postprocessing.vtk.read_vtk_data(
        vtk_filepath = str(datadir.join("water_freezing_endtime.vtu")))
        
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
    