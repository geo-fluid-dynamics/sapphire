import firedrake as fe 
import fempy.mms
import fempy.models.binary_alloy_enthalpy
import fempy.benchmarks.analytical_binary_alloy_solidification
import fempy.patches
import matplotlib.pyplot as plt
import numpy as np


class BinaryAlloySolidification(fempy.models.binary_alloy_enthalpy.Model):
    
    def __init__(self, meshsize):
        
        self.meshsize = meshsize
        
        self.cold_wall_temperature = fe.Constant(-1.)
        
        self.initial_temperature = fe.Constant(0.)
        
        self.initial_concentration = fe.Constant(1.)
        
        super().__init__()
        
        self.update_initial_values()
        
        self.output_directory_path = self.output_directory_path.joinpath(
            "binary_alloy_solidification/")
        
    def init_mesh(self):
    
        self.mesh = fe.UnitIntervalMesh(self.meshsize)
        
    def update_initial_values(self):
    
        initial_values = fe.interpolate(
            fe.Expression(
                (self.initial_temperature.__float__(), 
                 self.initial_concentration.__float__()),
                element = self.element),
            self.function_space)
            
        for iv in self.initial_values:
        
            iv.assign(initial_values)
        
    def init_dirichlet_boundary_conditions(self):
    
        W = self.function_space
        
        self.dirichlet_boundary_conditions = [
            fe.DirichletBC(W.sub(0), self.cold_wall_temperature, 1),]
            
            
def run_binary_alloy_solidification(
        stefan_number, 
        lewis_number,
        liquidus_slope,
        solid_concentration,
        initial_temperature,
        cold_wall_temperature,
        endtime, 
        meshsize,
        timestep_size,
        smoothing):
    
    model = BinaryAlloySolidification(meshsize = meshsize)
    
    model.smoothing.assign(smoothing)
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "Ste" + str(stefan_number) 
        + "_Le" + str(lewis_number) 
        + "mL" + str(liquidus_slope)
        + "_Tinf" + str(initial_temperature) + "_TB" 
        + str(cold_wall_temperature) + "/")
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "nx" + str(meshsize) 
        + "_Deltat" + str(timestep_size)
        + "_s" + str(smoothing) + "/")
    
    model.output_directory_path.mkdir(parents = True, exist_ok = True)
    
    model.stefan_number.assign(stefan_number)
    
    model.lewis_number.assign(lewis_number)
    
    model.liquidus_slope.assign(liquidus_slope)
    
    model.solid_concentration.assign(solid_concentration)
    
    model.cold_wall_temperature.assign(cold_wall_temperature)
    
    model.initial_temperature.assign(initial_temperature)
    
    model.initial_concentration.assign(1.)
    
    model.update_initial_values()
    
    model.solution.assign(model.initial_values[0])
    
    model.timestep_size.assign(timestep_size)
    
    
    V = fe.FunctionSpace(
            model.mesh, fe.FiniteElement("P", model.mesh.ufl_cell(), 1))
    
    figures = []
    
    axes = []
    
    for i in range(4):
    
        figures.append(plt.figure())
        
        axes.append(plt.axes())
    
    legend_strings = []
    
    times_to_plot = (endtime/4., endtime/2., 3.*endtime/4., endtime)
    
    for time, color in zip(times_to_plot, ("k", "r", "g", "b")):
    
        model.run(endtime = time, plot = False)
        
        legend_strings.append(r"$t = " + str(model.time.__float__()) + "$")
    
        T, Cl = model.solution.split()
        
        _phil = model.porosity(T, Cl)
        
        phil = fe.interpolate(_phil, V)
        
        C = fe.interpolate(_phil*Cl, V)
        
        for u_h, fig, ax, name, label in zip(
                (T, Cl, phil, C),
                figures,
                axes,
                ("T", "Cl", "phil", "C"),
                ("T", "C_l", "\\phi_l", "C")):
                
            plt.figure(fig.number)
            
            fempy.patches.plot(
                u,
                sample_points = sample_points,
                axes = ax,
                color = color,
                linestyle = "-")
            
            ax.set_xlabel(r"$x$")
            
            ax.set_ylabel(r"$" + str(label) + "$")
            
            ax.set_aspect(1./ax.get_data_ratio())
            
            ax.legend(legend_strings)
            
            filepath = model.output_directory_path.\
                joinpath(name).with_suffix(".png")
            
            print("Writing plot to " + str(filepath))
            
            fig.savefig(str(filepath), bbox_inches = "tight")
    
    
def scale_saline_freezing():

    L = 0.038  # side length of experimental cavity in @cite{michalek2003} [m]
    
    
    """ Set reference values for pure water thermodynamic properties per @cite{michalek2003}.
    Mostly these seem right for being near the freezing temperature.
    """
    k = 0.6  # Thermal conductivity [W/(m*K)]
    
    rho = 999.8  # Density [kg/m^3]
    
    c_p = 4182.  # Specific heat capacity [J/(kg*K)]
    
    h_m = 335000.  # Specific latent heat [J/kg]
    
    T_m = 0.  # Melting temperature of pure water-ice [deg C]
    
    
    # Set material properties for salt water as an eutectic binary alloy.
    T_E = -21.1  # Eutectic point temperature [deg C]
    
    C_E = 23.3  # Eutectic point concentration [wt. % NaCl]
    
    Le = 80.  # Lewis number
    
    C_s = 0.  # Solute concentration in the solid
    
    
    # Define linear liquidus
    def T_L(C):
    
        return T_m + (T_E - T_m)/C_E*C_0
        
    
    # Set typical sea water values
    C_0 = 3.5  # Salt concentration [wt. % NaCl]
    
    
    # Set initial and boundary values for the problem.
    """ Set boundary temperature to the eutectic temperature. """
    T_B = T_E  # Cold wall temperature [deg C]
    
    
    """ Set farfield temperature to liquidus given the farfield concentration. """    
    T_inf = T_L(C_0)  # Initial/farfield temperature [deg C]
    
    print("T_inf = " + str(T_inf))
    
    # Compute derived material properties.
    alpha = k/(rho*c_p)
    
    D = alpha/Le
    
    print("alpha = " + str(alpha))
    
    
    Ste = c_p*(T_inf - T_B)/h_m
    
    print("Stefan number = " + str(Ste))
    
    
    T_ref = T_L(C_0)
    
    def theta(T):
        
        return (T - T_ref)/(T_inf - T_B)
    
    theta_B = theta(T_B)
    
    print("theta_B = " + str(theta_B))
    
    theta_inf = theta(T_inf)
    
    print("theta_inf = " + str(theta_inf))
    
    
    def t_sec(t):
    
        return pow(L, 2)*t/alpha
        
    return Ste, Le, C_s, t_sec, theta_inf, theta_B
    
    
def run_saline_freezing(meshsize, timesteps, smoothing):
    """ Run the binary alloy solidification model 
    with saline freezing parameters.
    
    Note: Stability of the nonlinear solver is seeming to require 
    quadrupling the number of time steps when doubling the grid size.
    This would be quite impractical, 
    so we should test the behavior more rigorously.
    """
    Ste, Le, C_s, t_sec, theta_inf, theta_B = scale_saline_freezing()
    
    endtime = 1.
    
    print("End time in seconds = " + str(t_sec(endtime)))
    
    u_h = run_binary_alloy_solidification(
        stefan_number = Ste,
        lewis_number = Le,
        solid_concentration = C_s,
        endtime = endtime,
        initial_temperature = theta_inf,
        cold_wall_temperature = theta_B,
        meshsize = meshsize,
        timestep_size = endtime/float(timesteps),
        smoothing = smoothing)
        
    
def test__saline_freezing():
    
    run_saline_freezing(meshsize = 512, timesteps = 64, smoothing = 1./32.)
    
    # @todo: Compare to analytical solution
    
    
def test__binary_alloy_solidification():
    
    run_binary_alloy_solidification(
        stefan_number = 0.2,
        lewis_number = 10.,
        solid_concentration = 0.,
        initial_temperature = 0.5,
        cold_wall_temperature = -0.5,
        endtime = 1.,
        meshsize = 512,
        timestep_size = 1./64.,
        smoothing = 1./32.)
    
    
def compare_bas_to_analytical_solution(
        length_scale,
        thermal_conductivity,
        density,
        specific_heat_capacity,
        specific_latent_heat,
        pure_melting_temperature,
        eutectic_temperature,
        eutectic_concentration,
        lewis_number,
        solid_concentration,
        initial_temperature,
        initial_concentration,
        cold_wall_temperature,
        simulated_endtime,
        meshsize,
        simulated_timestep_size,
        smoothing,
        sample_size = 1000):
    
    
    L = length_scale
    
    k = thermal_conductivity
    
    rho = density
    
    c_p = specific_heat_capacity
    
    h_m = specific_latent_heat
    
    T_m = pure_melting_temperature
    
    T_E = eutectic_temperature
    
    C_E = eutectic_concentration
    
    Le = lewis_number
    
    C_s = solid_concentration
    
    C_0 = initial_concentration
    
    
    T_B = cold_wall_temperature
    
    T_inf = initial_temperature
    
    
    # Compute derived material properties.
    alpha = k/(rho*c_p)
    
    D = alpha/Le
    
    
    # Compute the analytical model
    _C, _T, x_s, _, _ = fempy.benchmarks.\
        analytical_binary_alloy_solidification.solve(
            k = k,
            rho = rho, 
            c_p = c_p, 
            h_m = h_m, 
            T_m = T_m, 
            T_B = cold_wall_temperature,
            T_inf = initial_temperature,
            D = D,
            C_0 = initial_concentration,
            T_E = T_E,
            C_E = C_E)
    
    
    # Scale values
    Ste = c_p*(T_inf - T_B)/h_m
    
    stefan_number = Ste
    
    T_ref = T_inf
    
    def chi(x):
    
        return L*x
    
    def theta(T):
        
        return (T - T_ref)/(T_inf - T_B)
    
    def xi(C):
    
        return C/initial_concentration
    
    m_L = (theta(T_E) - theta(T_m))/xi(C_E)
    
    liquidus_slope = m_L
    
    model = BinaryAlloySolidification(meshsize = meshsize)
    
    model.smoothing.assign(smoothing)
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "Ste" + str(stefan_number) 
        + "_Le" + str(lewis_number) 
        + "mL" + str(liquidus_slope)
        + "_thetainf" + str(theta(T_inf)) 
        + "_thetaB" + str(theta(T_B))
        + "/")
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "nx" + str(meshsize) 
        + "_Deltatau" + str(simulated_timestep_size)
        + "_s" + str(smoothing) 
        + "/")
        
    model.output_directory_path = model.output_directory_path.joinpath(
        "tauf" + str(simulated_endtime) + "/")
    
    model.output_directory_path.mkdir(parents = True, exist_ok = True)
    
    model.stefan_number.assign(stefan_number)
    
    model.lewis_number.assign(lewis_number)
    
    model.liquidus_slope.assign(liquidus_slope)
    
    model.solid_concentration.assign(xi(C_s))
    
    model.cold_wall_temperature.assign(theta(T_B))
    
    model.initial_temperature.assign(theta(T_inf))
    
    model.initial_concentration.assign(xi(C_0))
    
    model.update_initial_values()
    
    model.solution.assign(model.initial_values[0])
    
    model.timestep_size.assign(simulated_timestep_size)
    
    
    V = fe.FunctionSpace(
            model.mesh, fe.FiniteElement("P", model.mesh.ufl_cell(), 1))
    
    figures = []
    
    axes = []
    
    for i in range(4):
    
        figures.append(plt.figure())
        
        axes.append(plt.axes())
    
    legend_strings = []
    
    tau_f = simulated_endtime
    
    times_to_plot = (tau_f/4., tau_f/2., 3.*tau_f/4., tau_f)
    
    sample_points = [x/float(sample_size) for x in range(sample_size + 1)]
    
    def _phil(t, x):
        """ The analytical model does not include a mushy layer. """
        if x <= x_s(t):
        
            return 0.
            
        else:
        
            return 1.
    
    def _Cl(t, x):
        """ $C = C_l*\phi_l + C_s*(1 - \phi_l)$"""
        return (_C(t, x) - C_s*(1. - _phil(t, x)))/_phil(t, x)
        
    def _theta(t, x):
    
        return theta(_T(t, x))
    
    def _xil(t, x):
    
        return xi(_Cl(t, x))
        
    def _xi(t, x):
    
        return xi(_C(t, x))
    
    def t(tau):
    
        return tau*pow(L, 2)/alpha
    
    colors = [plt.cm.cool(i) for i in np.linspace(0, 1, len(times_to_plot))]
    
    for tau, color in zip(times_to_plot, colors):
    
        model.run(endtime = tau, plot = False)
        
        legend_strings.append(r"$u, \tau = " + str(model.time.__float__()) + "$")
        
        legend_strings.append(r"$u_h, \tau = " + str(model.time.__float__()) + "$")
    
        theta_h, xil_h = model.solution.split()
        
        _phil_h = model.porosity(theta_h, xil_h)
        
        phil_h = fe.interpolate(_phil_h, V)
        
        xi_h = fe.interpolate(_phil_h*xil_h, V)
        
        names_for_file = ("theta", "xil", "phil", "xi")
        
        yaxis_labels = (
            r"\theta \equiv " +
                r"\left(T - T_{L,0}\right)/\left(T_{L,0} - T_B\right)",
            r"\xi_l \equiv C_l/C_{l,0}", 
            r"\phi_l", 
            r"\xi \equiv C/C_0")
        
        for u_h, u, fig, ax, name, label in zip(
                (theta_h, xil_h, phil_h, xi_h),
                (_theta, _xil, _phil, _xi),
                figures,
                axes,
                names_for_file,
                yaxis_labels):
                
            plt.figure(fig.number)
            
            lines = plt.plot(
                sample_points, 
                [u(t(tau), L*p) for p in sample_points],
                axes = ax,
                color = color)
            
            lines[-1].set_linestyle("-")
            
            lines = fempy.patches.plot(
                u_h, 
                sample_points = sample_points,
                axes = ax,
                color = color)
                
            lines[-1].set_linestyle("--")
            
            ax.set_xlabel(r"$x/L$")
            
            ax.set_ylabel(r"$" + str(label) + "$")
            
            ax.set_aspect(1./ax.get_data_ratio())
            
            ax.legend(legend_strings)
            
            filepath = model.output_directory_path.\
                joinpath(name).with_suffix(".png")
            
            print("Writing plot to " + str(filepath))
            
            fig.savefig(str(filepath), bbox_inches = "tight")

  
def test__verify_bas_without_supercooling_against_analytical_solution():

    L = 0.038  # side length of experimental cavity in @cite{michalek2003} [m]
    
    
    """ Set reference values for pure water thermodynamic properties per @cite{michalek2003}.
    Mostly these seem right for being near the freezing temperature.
    """
    k = 0.6  # Thermal conductivity [W/(m*K)]
    
    rho = 999.8  # Density [kg/m^3]
    
    c_p = 4182.  # Specific heat capacity [J/(kg*K)]
    
    T_m = 0.  # Melting temperature of pure water-ice [deg C]
    
    h_m = 335000.  # Specific latent heat [J/kg]
    
    
    # Set material properties for salt water as an eutectic binary alloy.
    T_E = -21.1  # Eutectic point temperature [deg C]
    
    C_E = 23.3  # Eutectic point concentration [wt. % NaCl]
    
    C_s = 0.  # Solute concentration in the solid
    
    Le = 80.  # Lewis number
    
    
    # Set typical sea water values
    C_0 = 3.5  # Salt concentration [wt. % NaCl]
    
    
    # Set initial and boundary values for the problem.
    """ Set boundary temperature to the eutectic temperature. """
    T_B = T_E  # Cold wall temperature [deg C]
    
    """ Set farfield temperature to liquidus given the farfield concentration. """
    def T_L(C):
    
        return T_m + (T_E - T_m)/C_E*C_0
        
    T_inf = T_L(C_0)  # Initial/farfield temperature [deg C]
    
    
    # Try a low Le value to prevent supercooling
    low_Le = 2.
    
    #
    compare_bas_to_analytical_solution(
        length_scale = L,
        thermal_conductivity = k,
        density = rho,
        specific_heat_capacity = c_p,
        specific_latent_heat = h_m,
        pure_melting_temperature = T_m,
        eutectic_temperature = T_E,
        eutectic_concentration = C_E,
        lewis_number = low_Le,
        solid_concentration = C_s,
        initial_concentration = C_0,
        initial_temperature = T_inf,
        cold_wall_temperature = T_B,
        simulated_endtime = 1./8.,
        meshsize = 512,
        simulated_timestep_size = 1./64.,
        smoothing = 1./4096.)
    
    
class VerifiableModel(fempy.models.binary_alloy_enthalpy.Model):
    
    def __init__(self, meshsize):
    
        self.meshsize = meshsize
        
        super().__init__()
        
    def init_mesh(self):
    
        self.mesh = fe.UnitIntervalMesh(self.meshsize)
        
    def init_integration_measure(self):

        self.integration_measure = fe.dx(degree = 4)
        
    def strong_form_residual(self, solution):
        
        T, Cl = solution
        
        t = self.time
        
        Ste = self.stefan_number
        
        Le = self.lewis_number
        
        Cs = self.solid_concentration
        
        phil = self.porosity(T, Cl)
        
        diff, div, grad = fe.diff, fe.div, fe.grad
        
        r_T = diff(T, t) - div(grad(T)) + 1./Ste*diff(phil, t)
        
        r_Cl = phil*diff(Cl, t) - 1./Le*div(phil*grad(Cl)) + \
            (Cl - Cs)*diff(phil, t)
        
        return r_T, r_Cl
    
    def init_manufactured_solution(self):
        
        x = fe.SpatialCoordinate(self.mesh)[0]
        
        t = self.time
        
        sin, pi, exp = fe.sin, fe.pi, fe.exp
        
        T = 0.5*sin(2.*pi*x)*(1. - 2*exp(-3.*t**2))
        
        Cs = self.solid_concentration
        
        Cl = 0.5 + Cs - T
        
        self.manufactured_solution = T, Cl

    def init_initial_values(self):
        
        self.initial_values = fe.Function(self.function_space)
        
        for u_m, V in zip(
                self.manufactured_solution, self.function_space):
        
            self.initial_values.assign(fe.interpolate(u_m, V))
            

def test__fails__verify_spatial_convergence_order_via_mms(
        mesh_sizes = (4, 8, 16, 32, 64),
        timestep_size = 1./256.,
        tolerance = 0.1,
        plot_errors = False,
        plot_solution = False):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableModel,
        parameters = {
            "stefan_number": 0.1,
            "lewis_number": 8.,
            "solid_concentration": 0.02,
            "smoothing": 1./32.},
        expected_order = 2,
        mesh_sizes = mesh_sizes,
        timestep_size = timestep_size,
        endtime = 1.,
        tolerance = tolerance,
        plot_errors = plot_errors,
        plot_solution = plot_solution)
        