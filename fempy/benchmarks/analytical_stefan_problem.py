""" This module solves the analytical 1D Stefan problem """
import numpy as np
import scipy
import scipy.special
import scipy.optimize
import matplotlib.pyplot as plt


def solve(k, rho, c_p, h_m, T_m, T_B, T_inf, D, C_0, T_E, C_E):
    """ Solves 1D Stefan problem for the given parameters.
    This solves equations (2.6ab,2.7,2.8,2.9) from the 
    "Solidification of Fluids" chapter in
       @misc{batchelor2000perspectives,
       title={Perspectives in Fluid Dynamics},
       author={Batchelor, GK and Moffatt, HK and Worster, M},
       year={2000},
       publisher={Cambridge University Press Cambridge}
    
    
    Parameters
    ----------
    k
        thermal conductivity
    rho
        density
    c_p
        specific heat capacity
    h_m
        specific latent heat
    T_m
        melting temperature
    T_B
        cooled boundary temperature
    T_inf
        farfield temperature
       
    Returns
    -------
    T
        temperature function def T(t, x)
    h
        PCI position function def h(t)
    """
    pi, sqrt, exp = np.pi, np.sqrt, np.exp
    
    erf = scipy.special.erf
    
    
    alpha = k/(rho*c_p)  # thermal diffusivity
    
    Ste = c_p*(T_m - T_B)/h_m  # Stefan Number, Eq 2.9
    
    
    def eta(t, x):
    
        return x/(2.*sqrt(alpha*t))  # Eq 4.11
    
    
    
    
    def eq26(_lambda, Ste):

        return _lambda*exp(pow(_lambda, 2))*erf(_lambda) - Ste/sqrt(pi)
    
    _lambda = scipy.optimize.fsolve(G - Ste, sqrt(Ste/2.)
    
    
    def h(t):
        
        return 2.*_lambda*sqrt(alpha*t)  # Eq 4.7, position of the PCI
    
    def T(t, x):
        
        return T_B + (T_m - T_B)*erf(eta(t, x))/erf(_lambda)
    
    return T, h
    
    
def run_octadecane_example():
    """ Run an octadecane melting example. """

    
    # Set material properties of water-ice per @cite{lide2010}
    k = 2.14  # Thermal conductivity of the solid [W/(m*K)]
    
    rho = 916.7  # Density [kg/m^3]
    
    c_p = 2110.  # Heat capacity of the liquid [W/(m*K)]
    
    h_m = 333641.9   # Latent heat [J/kg]
    
    T_m = 0.  # Freezing temperature of pure water [deg C]
    
    
    # Set material properties for salt water as an eutectic binary alloy
    # per @cite{worster2000}
    T_E = -21.1  # Eutectic point temperature [deg C]
    
    C_E = 23.3  # Eutectic point concentration [wt. % NaCl]
    
    Le = 80.  # Lewis number
    
    
    # Set initial and boundary values for the example problem.
    """ Set boundary temperature to the eutectic temperature. """
    T_B = T_E  # Cold wall temperature [deg C]
    
    C_0 = 3.5  # Concentration [wt. % NaCl]
    
    """ Set farfield temperature to liquidus given the farfield concentration. """
    T_inf = T_E/C_E*C_0  # Initial/farfield temperature [deg C]
    
    
    # Compute derived material properties.
    alpha = k/(rho*c_p)
    
    D = alpha/Le
    
    
    # Get the analytical Stefan problem solution per (Worster 2000)
    C_fun, T_fun, h_fun, T_i, C_i = solve(
        k = k, rho = rho, c_p = c_p, h_m = h_m, T_m = T_m, 
        T_B = T_B, T_inf = T_inf, D = D, C_0 = C_0, T_E = T_E, C_E = C_E)
    
    
    # Evaluate the solution at some discrete times and discrete points
    L = 0.02
    
    resolution = 1000
    
    times_to_plot = (10., 20., 30.)  # [s]
    
    colors = ("r", "g", "b")
    
    
    assert(len(times_to_plot) == len(colors))
    
    
    x = np.linspace(0., L, resolution)
    
    
    fig = plt.figure()
    
    
    Cax = fig.add_subplot(111)
    
    Cax.set_ylim((-0.05*C_E, 1.05*C_E))
    
    Cax.set_yticks((0., C_0, C_E))
    
    Cax.set_ylabel(r"$C$ [wt. % NaCl]")
    
    Cax.set_xlabel(r"$x$ [m]")
    
    Cax.set_xlim([-L/20., 1.05*L]) 

    Cax.set_xticks(np.linspace(0., L, 5))
    
    
    Tax = fig.add_subplot(111, sharex=Cax, frameon=False)
    
    Tax.set_ylim((1.05*T_E, T_inf + (T_inf - T_E)*0.05))
    
    Tax.yaxis.tick_right()
    
    Tax.set_yticks((T_E, T_B, T_inf))
    
    Tax.set_ylabel(r"$T$ [$^\circ$C]")
    
    Tax.yaxis.set_label_position("right")
    
    
    T_legend_strings = []
    
    C_legend_strings = []
    
    
    for t, color in zip(times_to_plot, colors):

        h = h_fun(t)
        
        C = C_fun(t, x)
        
        T = T_fun(t, x)
        
        Cax.plot(x, C, color + "-")
        
        C_legend_strings.append("$C$ @ $t = " + str(t) + "$ s")
        
        Tax.plot(x, T, color + "--")
        
        T_legend_strings.append("$T$ @ $t = " + str(t) + "$ s")
        
        
    Cax.legend(C_legend_strings, loc = "center", framealpha = 0.)
    
    Tax.legend(T_legend_strings, loc = "center right", framealpha = 0.)
    
    
    plt.tight_layout()
    
    plt.savefig("saline_freezing.png")
    
    
if __name__ == "__main__":

    run_saline_example()
    