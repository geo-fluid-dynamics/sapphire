import firedrake as fe 
import fempy.mms
import fempy.models.binary_alloy_enthalpy
import fempy.benchmarks.analytical_binary_alloy_solidification
import matplotlib.pyplot as plt
import fempy.patches


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
            fe.DirichletBC(W.sub(0), self.cold_wall_temperature, 1),
            fe.DirichletBC(W.sub(0), self.initial_temperature, 2)]
            
            
def run_binary_alloy_solidification(
        stefan_number, 
        lewis_number, 
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
        + "_Tinf" + str(initial_temperature) + "_TB" 
        + str(cold_wall_temperature) + "/")
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "nx" + str(meshsize) 
        + "_Deltat" + str(timestep_size)
        + "_s" + str(smoothing) + "/")
    
    model.output_directory_path.mkdir(parents = True, exist_ok = True)
    
    model.stefan_number.assign(stefan_number)
    
    model.lewis_number.assign(lewis_number)
    
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
        
        for u, fig, ax, name, label in zip(
                (T, Cl, phil, C), 
                figures,
                axes,
                ("T", "Cl", "phil", "C"),
                ("T", "C_l", "\\phi_l", "C")):
                
            plt.figure(fig.number)
            
            fempy.patches.plot(u, axes = ax, color = color)
            
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
    
    
    # Set typical sea water values
    C_0 = 3.5  # Salt concentration [wt. % NaCl]
    
    
    # Set initial and boundary values for the problem.
    """ Set boundary temperature to the eutectic temperature. """
    T_B = T_E  # Cold wall temperature [deg C]
    
    """ Set farfield temperature to liquidus given the farfield concentration. """
    T_inf = T_m + T_E/C_E*C_0  # Initial/farfield temperature [deg C]
    
    print("T_inf = " + str(T_inf))
    
    # Compute derived material properties.
    kappa = k/(rho*c_p)
    
    D = kappa/Le
    
    alpha = k/(rho*c_p)
    
    print("alpha = " + str(alpha))
    
    
    Ste = c_p*(T_inf - T_B)/h_m
    
    print("Stefan number = " + str(Ste))
    
    
    T_ref = T_inf
    
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
    
    run_binary_alloy_solidification(
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
        stefan_number = 1.,
        lewis_number = 1.,
        solid_concentration = 0.,
        initial_temperature = 0.5,
        cold_wall_temperature = -0.5,
        endtime = 1./8.,
        meshsize = 512,
        timestep_size = 1./64.,
        smoothing = 1./32.)
    
    # @todo: Compare to analytical solution

    
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
        