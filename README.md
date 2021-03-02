![Sapphire](https://github.com/geo-fluid-dynamics/sapphire-docs/blob/master/Sapphire-Logo.png?raw=true)

Sapphire (mnemonically: Simulations automatically programmed with [Firedrake](https://www.firedrakeproject.org/)) 
is an engine for constructing PDE-based simulations 
discretized in space with mixed finite elements
and in time with finite differences.

[![Build Status](https://travis-ci.org/geo-fluid-dynamics/sapphire.svg?branch=master)](https://travis-ci.org/geo-fluid-dynamics/sapphire)
[![DOI](https://zenodo.org/badge/157389237.svg)](https://zenodo.org/badge/latestdoi/157389237)


## Examples

Details from the following examples were published in a [journal article](https://doi.org/10.1016/j.camwa.2020.11.008).

### Melting gallium
<img src="https://github.com/geo-fluid-dynamics/sapphire-docs/blob/master/GalliumMelting.gif?raw=true" height="400" />

### Freezing water
<img src="https://github.com/geo-fluid-dynamics/sapphire-docs/blob/master/WaterFreezing.gif?raw=true" height="400" />

### Melting octadecane
<img src="https://github.com/geo-fluid-dynamics/sapphire-docs/blob/master/OctadecaneMelting.gif?raw=true" height="400" />

## Setup

### Firedrake
[Install Firedrake](https://www.firedrakeproject.org/download.html).

Activate the Firedrake virtual environment with something like

    . ~/firedrake/bin/activate
    

### Sapphire
Download with 

    git clone git@github.com:geo-fluid-dynamics/sapphire.git

The following assumes that the Firedrake virtual environment is already activated.

Test with

    python3 -m pytest sapphire/tests/

Install with

    cd sapphire
    
    python3 setup.py install
    
    
## Development

### Guidelines
Mostly we try to follow PEP proposed guidelines, e.g. [The Zen of Python (PEP 20)](https://www.python.org/dev/peps/pep-0020/), and do not ever `from firedrake import *` ([PEP 8](https://www.python.org/dev/peps/pep-0008/)).

This package structure mostly follows that suggested by [The Hitchhiker's Guide to Python](http://docs.python-guide.org/en/latest/).
