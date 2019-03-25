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
    
    fig, _ = fempy.postprocessing.vtk.plot_mesh(solution_filepath = \
        str(datadir.join("mixed_vector_valued_solution.vtu")))
    
    outpath = datadir.join("mesh.png")
    
    print("Saving {0}".format(outpath))
    
    fig.savefig(outpath)
    