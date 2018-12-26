""" This module solves the analytical 1D Stefan problem for binary alloy solidification. """
import numpy
import scipy
import scipy.special
import scipy.optimize
import matplotlib.pyplot as plt


def solve(k, rho, C_p, L, T_m, T_B, T_inf, D, C_0, T_E, C_E):
    """ Solves 1D binary alloy solidification for the given parameters.
    This solves equations 2.8, 2.14, 4.6-4.12, 4.14 from the 
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
    C_p
        specific heat capacity
    L
        latent heat
    T_m
        melting temperature
    T_B
        cooled boundary temperature
    T_inf
        farfield temperature
    D
        mass diffusivity
    C_0
        initial/farfield concentration
    T_E
        eutectic temperature
    C_E
        eutectic concentration
        
    Returns
    -------
    C
        concentration function def C(t, z)
    T
        temperature function def T(t, z)
    h
        PCI position function def h(t)
    T_i
        the interfacial temperature
    C_i
        the interfacial concentration
    """
    pi, sqrt, exp = numpy.pi, numpy.sqrt, numpy.exp
    
    erf, erfc = scipy.special.erf, scipy.special.erfc
    
    
    kappa = k/(rho*C_p)  # thermal diffusivity
    
    Le = kappa/D  # Lewis number
    
    epsilon = 1./sqrt(Le)  # Worster uses this in place of the Lewis number
    
    m = (T_m - T_E)/C_E  # Slope of ideal liquidius curve, Figure 5
    
    
    def G(_lambda):

        return sqrt(pi)*_lambda*exp(_lambda**2)*erf(_lambda)  # Eq 2.8
    
    def F(_lambda):

        return sqrt(pi)*_lambda*exp(_lambda**2)*erfc(_lambda)  # Eq 2.14
    
    def Equation4p12ab(_lambda):

        return (T_m + m*C_0/(F(_lambda) - 1.) - T_B)/G(epsilon*_lambda) - \
        (T_inf - T_m - m*C_0/(F(_lambda) - 1.))/F(epsilon*_lambda) - L/C_p
        # Equation 4.12b sub 4.6 sub 4.12a
    
    _lambda = scipy.optimize.fsolve(Equation4p12ab, 0.1)
    
    C_i = -C_0/(F(_lambda) - 1.)  # Eq 4.12a
    
    def eta(t, z):
    
        return z/(2.*sqrt(D*t))  # Eq 4.11
    
    
    def h(t):
        
        return 2.*_lambda*sqrt(D*t)  # Eq 4.7, position of the PCI
    
    def T(t, z):
    
        TL = (eta(t,z) < _lambda)* \
            (T_B + (T_m - m*C_i - T_B)*erf(epsilon*eta(t, z))/erf(epsilon*_lambda))
            # Equation 4.8 sub 4.6
        
        TR = (eta(t,z) > _lambda)* \
            (T_inf + (T_m - m*C_i - T_inf)*erfc(epsilon*eta(t,z))/erfc(epsilon*_lambda))
            # Equation 4.9 sub 4.6
        
        return TL + TR
    
    def C(t, z):
        
        return (eta(t,z) > _lambda)*(C_0 + (C_i - C_0)*erfc(eta(t,z))/erfc(_lambda))  # Eq 4.10
        
    T_i = T_m - m*C_i  # Eq 4.6
    
    return C, T, h, T_i, C_i
    
    
def run_salt_water_example():
    """ Run a salt water solidification example. """

    
    # Set material properties for salt water.
    k = 2.3  # Thermal conductivity of the solid [W/(m*K)]
    
    rho = 920.  # Density [kg/m^3]
    
    C_p = 4200.  # Heat capacity of the liquid [W/(m*K)]
    
    L = 333700.  # Latent heat [J/kg]
    
    T_m = 0.  # Freezing temperature of pure water [deg C]
    
    T_E = -21.1  # Eutectic point temperature [deg C]
    
    C_E = 23.3  # Eutectic point concentration [wt. % NaCl]
    
    Le = 80.  # Lewis number
    
    
    # Set initial and boundary values for the example problem.
    T_B = -30.  # Cold wall temperature [deg C]
    
    T_inf = -20.  # Initial/farfield temperature [deg C]
    
    C_0 = 10  # Concentration [wt. % NaCl]
    
    
    # Compute derived material properties.
    kappa = k/(rho*C_p)
    
    D = kappa/Le
    
    
    # Get the analytical Stefan problem solution per (Worster 2000)
    C_fun, T_fun, h_fun, T_i, C_i = solve(k = k, rho = rho, C_p = C_p, L = L, T_m = T_m, 
        T_B = T_B, T_inf = T_inf, D = D, C_0 = C_0, T_E = T_E, C_E = C_E)
    
    
    # Evaluate the solution at some discrete times and discrete points
    L = 0.02
    
    resolution = 1000
    
    times_to_plot = (60, 600, 6000)  # [s]
    
    colors = ("r", "g", "b")
    
    
    assert(len(times_to_plot) == len(colors))
    
    
    z = numpy.linspace(0., L, resolution)
    
    
    fig = plt.figure()
    
    
    Tax = fig.add_subplot(111)
    
    plt.xlabel("$x$ [m]")
    
    plt.ylabel("$T$ [$^\circ$C]")
    
    plt.ylim([T_B, -15.])
    
    
    Cax = fig.add_subplot(111, sharex=Tax, frameon=False)
    
    Cax.yaxis.tick_right()
    
    Cax.yaxis.set_label_position("right")
    
    plt.ylabel("$C$ [wt. % NaCl]")
    
    plt.ylim([0., 45.])
    
    
    T_legend_strings = []
    
    C_legend_strings = []
    
    
    for t, color in zip(times_to_plot, colors):

        h = h_fun(t)
        
        C = C_fun(t, z)
        
        T = T_fun(t, z)
        
        Tax.plot(z, T, color + "-")
        
        T_legend_strings.append("$T$ @ $t = " + str(t) + "$ s")
        
        Cax.plot(z, C, color + "--")
        
        C_legend_strings.append("$C$ @ $t = " + str(t) + "$ s")
        
    plt.xlim([0., L])    
    
    Tax.legend(T_legend_strings, loc = "upper left", framealpha = 0.)
    
    Cax.legend(C_legend_strings, loc = "upper right", framealpha = 0.)
    
    plt.savefig("salt_water_solidification.png")
    
    
if __name__ == "__main__":

    run_salt_water_example()
    