import firedrake as fe
import fempy.models.binary_alloy_enthalpy_porosity
import fempy.applications.binary_alloy_cavity_freezing

    
def test__sea_ice_cavity_freezing__regression():

    endtime = 1.
    
    timestep_size = 1./4.
    
    meshsize = 32
    
    latent_heat_smoothing = 1./64.
    
    expected_liquid_area = 0.92
    
    tolerance = 0.01
    
    
    model = fempy.applications.binary_alloy_cavity_freezing.Model(
        meshsize = meshsize)
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "sea_ice_cavity_freezing/")
        
    
    model.hot_wall_temperature.assign(1./3.)
    
    model.cold_wall_temperature_before_freezing.assign(0.)
    
    model.cold_wall_temperature_during_freezing.assign(-2./3.)
    
    model.prandtl_number.assign(13.)
    
    model.schmidt_number.assign(1100.)
    
    model.stefan_number.assign(0.19)
    
    model.pure_liquidus_temperature.assign(0.)
    
    model.initial_concentration.assign(1.)
    
    model.liquidus_slope.assign(-0.1)
    
    """ Run with reduced Rayleigh numbers.
    With the other parameters in this test, 
    realistic Rayleigh numbers are Ra_T = 5.e6 and Ra_C = -6.e7;
    but running higher Rayleigh numbers may require stabilization.
    """
    model.temperature_rayleigh_number.assign(1.e6)
    
    model.concentration_rayleigh_number.assign(-1.e6)
    
    
    model.latent_heat_smoothing.assign(latent_heat_smoothing)
    
    model.timestep_size.assign(timestep_size)
    
    
    model.run(endtime = endtime, plot = False)
    
    
    p, u, T, Cl = model.solution.split()
    
    phil = model.porosity(T = T, Cl = Cl)
    
    liquid_area = fe.assemble(phil*fe.dx)
    
    print("Liquid area = " + str(liquid_area))
    
    assert(abs(liquid_area - expected_liquid_area) < tolerance)
    
    
class VerifiableModel(
    fempy.models.binary_alloy_enthalpy_porosity.Model):

    def __init__(self, meshsize):
        
        self.meshsize = meshsize
        
        super().__init__()
        
        self.temperature_rayleigh_number.assign(1.e5)
        
        self.concentration_rayleigh_number.assign(-1.e5)
        
        self.prandtl_number.assign(10.)
        
        self.stefan_number.assign(0.2)
        
        self.schmidt_number.assign(1.e3)
        
        self.pure_liquidus_temperature.assign(0.)
        
        self.liquidus_slope.assign(-0.1)
        
        self.latent_heat_smoothing.assign(1./32.)
        
    def init_mesh(self):
        
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def strong_form_residual(self, solution):
        
        gamma = self.pressure_penalty_factor
        
        mu_s = self.solid_dynamic_viscosity
        
        mu_l = self.liquid_dynamic_viscosity
        
        Ra = self.rayleigh_number
        
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        Sc = self.schmidt_number
        
        t = self.time
        
        grad, dot, div, sym, diff = fe.grad, fe.dot, fe.div, fe.sym, fe.diff
        
        p, u, T, Cl = solution
        
        b = self.buoyancy(T, Cl)
        
        phil = self.porosity(T, Cl)
        
        mu = mu_s + (mu_l - mu_s)*phil
        
        r_p = div(u) + gamma*p
        
        r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(mu*sym(grad(u))) + b
        
        r_T = diff(T, t) + div(T*u) - 1./Pr*div(grad(T)) + 1./Ste*diff(phil, t)
        
        r_Cl = phil*diff(Cl, t) + div(Cl*u) - 1./Sc*div(phil*grad(Cl)) \
            + Cl*diff(phil, t)
        
        return r_p, r_u, r_T, r_Cl
        
    def init_manufactured_solution(self):
        
        pi, sin, cos, exp = fe.pi, fe.sin, fe.cos, fe.exp
        
        x = fe.SpatialCoordinate(self.mesh)
        
        t = self.time
        
        t_f = fe.Constant(1.)
        
        ihat, jhat = self.unit_vectors()
        
        u = exp(t)*sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
            exp(t)*sin(pi*x[0])*sin(2.*pi*x[1])*jhat
        
        p = -0.5*(u[0]**2 + u[1]**2)
        
        T = 0.5*sin(2.*pi*x[0])*sin(pi*x[1])*(1. - 2*exp(-3.*t**2))
        
        Cl = sin(pi*x[0])*sin(2.*pi*x[1])*exp(-2.*t**2)
        
        self.manufactured_solution = p, u, T, Cl
        
    def init_initial_values(self):
        
        self.initial_values = fe.Function(self.function_space)
        
        for u_m, V in zip(
                self.manufactured_solution, self.function_space):
        
            self.initial_values.assign(fe.interpolate(u_m, V))
        
    def solve(self):
        
        self.solver.parameters["snes_monitor"] = False
        
        super().solve()
            
        print("Solved at time t = " + str(self.time.__float__()))
        
        
def fails__test__verify_spatial_convergence_order_via_mms(
        parameters = {
            "temperature_rayleigh_number": 10.,
            "concentration_rayleigh_number": 1.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "schmidt_number": 1.,
            "pure_liquidus_temperature": 0.,
            "liquidus_slope": -0.01,
            "latent_heat_smoothing": 1./16.},
        mesh_sizes = (4, 8, 16),
        timestep_size = 1./64.,
        tolerance = 0.2):
        
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableModel,
        parameters = parameters,
        expected_order = 2,
        mesh_sizes = mesh_sizes,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 1.)
    
    