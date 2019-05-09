import setuptools


setuptools.setup(
    name = 'sapphire',
    version = '0.4.3a0',
    packages = setuptools.find_packages(),
    install_requires = [
        'matplotlib',
        'numpy',
        'scipy',
        'vtk']]
)
