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
    
The scaling (and therefore also the Boussinesq buoyancy term) is different in \cite{belhamadia2019adaptive}.
They chose the reference time $t_r = \rho_l c_l x_r^2 / k_l$.
Their momentum equation therefore has $Pr$ in front of the momentum diffusion term
and their buoyancy equation for Gallium is $b(T) = Pr Ra T$.
We chose before $t_r = \nu_l / x^2_r$ which sets $Re = 1$.
The choice of $t_r = \rho_l c_l x_r^2 / k_l$ in \cite{belhamadia2019adaptive} sets $Re = 1/Pr$.
"""
import firedrake as fe
import sapphire.simulations.navier_stokes_boussinesq
import sapphire.simulations.examples.melt_octadecane


reference_length = 6.35  # cm

reference_temperature = 301.3  # K

reference_temperature_range = 9.7  # K

reference_time = 292.90  # s

class Simulation(sapphire.simulations.examples.melt_octadecane.Simulation):

    def __init__(self, *args,
            rayleigh_number = 7.e5,
            prandtl_number = 0.0216,
            stefan_number = 0.046,
            liquidus_temperature = 0.1525,
            hotwall_temperature = 1.,
            initial_temperature = 0.,
            cutoff_length = 0.5,
            element_degrees = (1, 2, 2),
            mesh_dimensions = (20, 40),
            **kwargs):
            
        if "solution" not in kwargs:
            
            mesh = fe.RectangleMesh(
                nx = mesh_dimensions[0],
                ny = mesh_dimensions[1],
                Lx = cutoff_length,
                Ly = 1.)
                
            element = sapphire.simulations.navier_stokes_boussinesq.element(
                cell = mesh.ufl_cell(),
                degrees = element_degrees)
                
            kwargs["solution"] = fe.Function(fe.FunctionSpace(mesh, element))
        
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
            