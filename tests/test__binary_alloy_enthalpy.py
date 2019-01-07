import firedrake as fe 
import fempy.mms
import fempy.models.binary_alloy_enthalpy
import fempy.benchmarks.analytical_binary_alloy_solidification
import matplotlib.pyplot as plt
import fempy.patches


class SalineFreezingModel(fempy.models.binary_alloy_enthalpy.Model):
    
    def __init__(self, meshsize):
        
        self.meshsize = meshsize
        
        self.cold_wall_temperature = fe.Constant(-1.)
        
        self.initial_temperature = fe.Constant(0.)
        
        self.initial_concentration = fe.Constant(1.)
        
        super().__init__()
        
        self.update_initial_values()
        
        self.output_directory_path = self.output_directory_path.joinpath(
            "saline_freezing/")
        
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
            
            
def fails__test__saline_freezing():

    # Set length scale
    L = 0.02  # [m]
    
    # Set material properties of water-ice per @cite{lide2010}
    k = 2.14  # Thermal conductivity [W/(m*K)]
    
    rho = 916.7  # Density [kg/m^3]
    
    c_p = 2110.  # Specific heat capacity [J/(kg*K)]
    
    h_m = 333641.9  # Specific latent heat [J/kg]
    
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
    
    endtime_sec = 30.  # [s]
    
    nx = 64
    """ Note: Stability of this method is seeming to require quadrupling
    the number of time steps when doubling the grid size.
    This would be quite impractical, 
    so we should test the behavior more rigorously.
    """
    nt = 12
    
    sinv = 128
    
    model = SalineFreezingModel(meshsize = nx)
    
    model.smoothing.assign(1./float(sinv))
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "nx" + str(nx) + "_nt" + str(nt) + "_sinv" + str(sinv) + "/")
    
    model.output_directory_path.mkdir(parents = True, exist_ok = True)
    
    Ste = c_p*(T_inf - T_B)/h_m
    
    print("Stefan number = " + str(Ste))
    
    model.stefan_number.assign(Ste)
    
    model.lewis_number.assign(Le)
    
    model.solid_concentration.assign(C_s)
    
    T_ref = T_inf
    
    def theta(T):
        
        return (T - T_ref)/(T_inf - T_B)

    def xi(C):
        
        return C/C_0
        
    theta_B = theta(T_B)
    
    print("theta_B = " + str(theta_B))
    
    model.cold_wall_temperature.assign(theta_B)
    
    theta_inf = theta(T_inf)
    
    model.initial_temperature.assign(theta_inf)
    
    print("theta_0 = " + str(theta_inf))
    
    xi_0 = xi(C_0)
    
    print("xi_0 = " + str(xi_0))
    
    model.initial_concentration.assign(xi_0)
    
    model.update_initial_values()
    
    model.solution.assign(model.initial_values[0])
    
    def t(t_sec):
    
        return t_sec*alpha/pow(L, 2)
    
    endtime = t(endtime_sec)
    
    print("Scaled end time = " + str(endtime))
    
    Delta_t = endtime/float(nt)
    
    model.timestep_size.assign(Delta_t)
    
    
    V = fe.FunctionSpace(
            model.mesh, fe.FiniteElement("P", model.mesh.ufl_cell(), 1))
    
    figures = []
    
    axes = []
    
    for i in range(4):
    
        figures.append(plt.figure())
        
        axes.append(plt.axes())
    
    legend_strings = []
    
    for time, color in zip(
            (endtime/3., 2.*endtime/3., endtime), ("r", "g", "b")):
    
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
            
            ax.legend(legend_strings)
        
            filepath = model.output_directory_path.\
                joinpath(name).with_suffix(".png")
        
            print("Writing plot to " + str(filepath))
    
            fig.savefig(str(filepath))
            
    assert(False)
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
            

def fails__test__verify_spatial_convergence_order_via_mms(
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
            "latent_heat_smoothing": 1./32.},
        expected_order = 2,
        mesh_sizes = mesh_sizes,
        timestep_size = timestep_size,
        endtime = 1.,
        tolerance = tolerance,
        plot_errors = plot_errors,
        plot_solution = plot_solution)
        