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
    
    fig, _ = fempy.postprocessing.vtk.plot_mesh(vtk_data = data)
    
    outpath = datadir.join("mesh.png")
    
    print("Saving {0}".format(outpath))
    
    fig.savefig(outpath)

    
def test__plot_scalar_field_contours(datadir):
    
    data = fempy.postprocessing.vtk.read_vtk_data(
        vtk_filepath = str(datadir.join("water_freezing_endtime.vtu")))
    
    
    for filled in (False, True):
    
        fig, _ = fempy.postprocessing.vtk.plot_field_contours(
            vtk_data = data,
            scalar_solution_component = 2,
            contours = 8,
            filled = filled)
        
        name = "temperature_contours"
        
        if filled:
        
            name += "_filled"
        
        outpath = datadir.join(name + ".png")
        
        print("Saving {0}".format(outpath))
    
        fig.savefig(outpath)
        
    
def test__plot_scalar_field(datadir):
    
    data = fempy.postprocessing.vtk.read_vtk_data(
        vtk_filepath = str(datadir.join("water_freezing_endtime.vtu")))
        
    fig, _ = fempy.postprocessing.vtk.plot_field(
        vtk_data = data,
        scalar_solution_component = 2)
    
    outpath = datadir.join("temperature.png")
    
    print("Saving {0}".format(outpath))
    
    fig.savefig(outpath)
    