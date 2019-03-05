# FemPy
FemPy is an engine for constructing PDE-based simulations 
discretized in space with mixed finite elements
and in time using finite differences.

[![DOI](https://zenodo.org/badge/157389237.svg)](https://zenodo.org/badge/latestdoi/157389237)


## Portfolio

### Water freezing
![Water freezing benchmark](https://github.com/geo-fluid-dynamics/fempy-docs/blob/master/WaterFreezing_Temperature_and_Velocity_16Colors.gif?raw=true)


## Setup
[Install Firedrake](https://www.firedrakeproject.org/download.html).

Activate the Firedrake virtual environment with something like

    . ~/firedrake/bin/activate
    
Download FemPy with 

    git clone git@github.com:geo-fluid-dynamics/fempy.git

The following assumes that the Firedrake virtual environment is already activated.

Test FemPy with

    python3 -m pytest fempy/

Install FemPy with

    python3 fempy/setup.py install
    
    
## Development

### Project structure
This project mostly follows the structure suggested by [The Hitchhiker's Guide to Python](http://docs.python-guide.org/en/latest/).


### Guidelines
Mostly we try to follow PEP proposed guidelines, e.g. [The Zen of Python (PEP 20)](https://www.python.org/dev/peps/pep-0020/), and do not ever `from firedrake import *` ([PEP 8](https://www.python.org/dev/peps/pep-0008/)).
