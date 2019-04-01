from pytest import fixture
from distutils import dir_util
import os


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
    