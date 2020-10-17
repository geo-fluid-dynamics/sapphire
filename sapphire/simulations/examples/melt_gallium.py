""" Gallium melting benchmark 

Physical parameters are based on 
    @article{belhamadia2019adaptive,
        author = {Belhamadia, Youssef and Fortin, André and Briffard, Thomas},
        year = {2019},
        month = {06},
        pages = {1-19},
        title = {A two-dimensional adaptive remeshing method for solving melting and solidification problems with convection},
        volume = {76},
        journal = {Numerical Heat Transfer, Part A: Applications},
        doi = {10.1080/10407782.2019.1627837},
    }

Belhamadia uses the temperature scaling $\tilde{T} = (T - T_r) / \Delta T$.
They chose reference temperature $T_r = 301.3 K$
and set nondimensional melting temperature $T_f = 0.1525$.
So their reference temperature was not chosen as the melting temperature.
They set the initial temperature to $T_c = 0$ and hot wall to $T_h = 1$.
This means that they chose $T_r = T_c$ and therefore $T_c = 301.3 K$.
The dimensional melting temperature is $0.1525*(9.7 K) + 301.3 K$ => $T_f = 302.8 K$.

We use the temperature scaling $\tilde{T} = (T - T_f)/ \Delta T$.
Therefore, $\tilde{T}_f = 0.$ and $\tilde{T}_c = -0.1546$.
"""
import firedrake as fe
import sapphire.simulations.navier_stokes_boussinesq
import sapphire.simulations.examples.melt_octadecane


reference_length = 6.35  # cm

coldwall_temperature = 301.3  # K

reference_temperature_range = 9.7  # K

reference_time = 292.90  # s



class Simulation(sapphire.simulations.examples.melt_octadecane.Simulation):

    def __init__(self, *args,
            rayleigh_number = 7.e5,
            prandtl_number = 0.0216,
            stefan_number = 0.046,
            liquidus_temperature = 0.,
            hotwall_temperature = 1.,
            initial_temperature = -0.1546,
            cutoff_length = 0.5,
            taylor_hood_pressure_degree = 1,
            temperature_degree = 2,
            mesh_dimensions = (20, 40),
            **kwargs):
            
        if "solution" not in kwargs:
            
            kwargs["mesh"] = fe.RectangleMesh(
                nx = mesh_dimensions[0],
                ny = mesh_dimensions[1],
                Lx = cutoff_length,
                Ly = 1.)
        
        super().__init__(
            *args,
            reynolds_number = 1./prandtl_number,
            rayleigh_number = rayleigh_number,
            prandtl_number = prandtl_number,
            stefan_number = stefan_number,
            liquidus_temperature = liquidus_temperature,
            hotwall_temperature = hotwall_temperature,
            initial_temperature = initial_temperature,
            **kwargs)
            