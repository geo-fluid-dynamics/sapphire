# Sunfire
Sunfire is an engine for constructing PDE-based simulations 
discretized in time with finite differences
and in space with mixed finite elements.

[![DOI](https://zenodo.org/badge/157389237.svg)](https://zenodo.org/badge/latestdoi/157389237)


## Portfolio

### Water freezing
![Water freezing benchmark](https://github.com/geo-fluid-dynamics/sunfire-docs/blob/master/WaterFreezing.gif?raw=true)


## Setup

### Firedrake
[Install Firedrake](https://www.firedrakeproject.org/download.html).

Activate the Firedrake virtual environment with something like

    . ~/firedrake/bin/activate
    

### Sunfire
Download with 

    git clone git@github.com:geo-fluid-dynamics/sunfire.git

The following assumes that the Firedrake virtual environment is already activated.

Test with

    python3 -m pytest sunfire/

Install with

    python3 sunfire/setup.py install
    
    
## Development

### Guidelines
Mostly we try to follow PEP proposed guidelines, e.g. [The Zen of Python (PEP 20)](https://www.python.org/dev/peps/pep-0020/), and do not ever `from firedrake import *` ([PEP 8](https://www.python.org/dev/peps/pep-0008/)).

This package structure mostly follows that suggested by [The Hitchhiker's Guide to Python](http://docs.python-guide.org/en/latest/).
