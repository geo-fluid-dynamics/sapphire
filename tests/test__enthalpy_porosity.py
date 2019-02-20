import firedrake as fe 
import fempy.mms
import fempy.models.enthalpy_porosity
import fempy.benchmarks.melting_octadecane

        
class SimpleResistanceModel(
        fempy.benchmarks.melting_octadecane.DarcyResistanceModel):
        
    def __init__(self, *args, **kwargs):
    
        super().__init__(*args, **kwargs)
        
        self.topwall_heatflux_postswitch = 0.
        
        self.topwall_heatflux_switchtime = 40. + 2.*self.time_tolerance
        
    def darcy_resistance(self, T):
        """ Resistance to flow based on permeability of the porous media """
        D = self.darcy_resistance_factor
        
        phil = self.porosity(T)
        
        return D*(1. - phil)
        
    def init_problem(self):
    
        self.topwall_heatflux = fe.Constant(0.)
        
        q = self.topwall_heatflux
        
        _, _, psi_T = fe.TestFunctions(self.function_space)
        
        ds = fe.ds(domain = self.mesh, subdomain_id = 4)
        
        r = self.weak_form_residual*self.integration_measure + psi_T*q*ds
        
        u = self.solution
        
        self.problem = fe.NonlinearVariationalProblem(
            r, u, self.dirichlet_boundary_conditions, fe.derivative(r, u))
        
    def solve(self):
    
        if self.time.__float__() >  \
                (self.topwall_heatflux_switchtime - self.time_tolerance):
            
            self.topwall_heatflux.assign(
                self.topwall_heatflux_postswitch)
    
        super().solve()


def test__regression__melting_octadecane_benchmark__heat_flux__validation__second_order():
    
    endtime = 80.
    
    topwall_heatflux_switchtime = 40.
    
    q = -0.02
    
    
    s = 1./64.
    
    nx = 32
    
    Delta_t = 10.
    
    D = 1.e12
    
    
    expected_liquid_area = 0.64
    
    tolerance = 0.01
    
    
    model = SimpleResistanceModel(
        quadrature_degree = 4, spatial_order = 2, temporal_order = 2, meshsize = nx)
    
    model.timestep_size.assign(Delta_t)
    
    model.smoothing.assign(s)
    
    model.darcy_resistance_factor.assign(D)
    
    model.topwall_heatflux_switchtime =  topwall_heatflux_switchtime +  \
        2.*model.time_tolerance
    
    model.topwall_heatflux_postswitch = q
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "melting_octadecane/heatflux_switchtime" 
        + str(topwall_heatflux_switchtime)
        + "_tf" + str(endtime) + "/"
        + "q" + str(q) + "/"
        + "second_order_"
        + "nx" + str(nx) + "_Deltat" + str(Delta_t) 
        + "_s" + str(s) + "_D" + str(D) + "/")
        
    model.run(endtime = endtime, plot = False, report = False)
    
        
    p, u, T = model.solution.split()
    
    phil = model.porosity(T)
    
    liquid_area = fe.assemble(phil*fe.dx)
    
    print("Liquid area = " + str(liquid_area))
    
    assert(abs(liquid_area - expected_liquid_area) < tolerance)


def test__melting_octadecane_benchmark__simple_resistance__second_order__regression():
    
    endtime, expected_liquid_area, tolerance = 30., 0.22, 0.01
    
    nx = 32
    
    Delta_t = 10.
    
    model = SimpleResistanceModel(
        quadrature_degree = 4, spatial_order = 2, temporal_order = 2, meshsize = nx)
    
    model.timestep_size.assign(Delta_t)
    
    s = 1./256.
    
    model.smoothing.assign(s)
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "melting_octadecane/simple_resistance_second_order/" + "nx" + str(nx) + "_Deltat" + str(Delta_t) 
        + "_s" + str(s) + "_tf" + str(endtime) + "/")
        
    model.run(endtime = endtime, plot = False, report = False)
    
    p, u, T = model.solution.split()
    
    phil = model.porosity(T)
    
    liquid_area = fe.assemble(phil*fe.dx)
    
    print("Liquid area = " + str(liquid_area))
    
    assert(abs(liquid_area - expected_liquid_area) < tolerance)
    
    phil_h = fe.interpolate(phil, model.function_space.sub(2))
    
    max_phil = phil_h.vector().max()
    
    print("Maximum phil = " + str(max_phil))
    
    assert(abs(max_phil - 1.) < tolerance)
        
        
def test__melting_octadecane_benchmark__viscosity__second_order__regression():
    
    endtime, expected_liquid_area, tolerance = 30., 0.21, 0.01
    
    nx = 32
    
    Delta_t = 10.
    
    model = fempy.benchmarks.melting_octadecane.Model(
        quadrature_degree = 4, spatial_order = 2, temporal_order = 2, meshsize = nx)
    
    model.timestep_size.assign(Delta_t)
    
    s = 1./256.
    
    model.smoothing.assign(s)
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "melting_octadecane/viscosity_second_order/" + "nx" + str(nx) + "_Deltat" + str(Delta_t) 
        + "_s" + str(s) + "_tf" + str(endtime) + "/")
        
    model.run(endtime = endtime, plot = False, report = False)
    
    p, u, T = model.solution.split()
    
    phil = model.porosity(T)
    
    liquid_area = fe.assemble(phil*fe.dx)
    
    print("Liquid area = " + str(liquid_area))
    
    assert(abs(liquid_area - expected_liquid_area) < tolerance)
    
    phil_h = fe.interpolate(phil, model.function_space.sub(2))
    
    max_phil = phil_h.vector().max()
    
    print("Maximum phil = " + str(max_phil))
    
    assert(abs(max_phil - 1.) < tolerance)
    
    
def test__melting_octadecane_benchmark__darcy__second_order__regression():
    
    endtime, expected_liquid_area, tolerance = 30., 0.21, 0.01
    
    D = 1.e12
    
    model = fempy.benchmarks.melting_octadecane.DarcyResistanceModel(
        quadrature_degree = 4, spatial_order = 2, temporal_order = 2, meshsize = 32)
    
    model.timestep_size.assign(10.)
    
    model.smoothing.assign(1./256.)
    
    model.darcy_resistance_factor.assign(D)
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "melting_octadecane_with_darcy_resistance/D" + str(D) + "/")
        
    model.run(endtime = endtime, plot = False, report = False)
    
    p, u, T = model.solution.split()
    
    phil = model.porosity(T)
    
    liquid_area = fe.assemble(phil*fe.dx)
    
    print("Liquid area = " + str(liquid_area))
    
    assert(abs(liquid_area - expected_liquid_area) < tolerance)
    
    phil_h = fe.interpolate(phil, model.function_space.sub(2))
    
    max_phil = phil_h.vector().max()
    
    print("Maximum phil = " + str(max_phil))
    
    assert(abs(max_phil - 1.) < tolerance)


class VerifiableModel(fempy.models.enthalpy_porosity.Model):

    def __init__(self,
            quadrature_degree,
            spatial_order,
            temporal_order,
            meshsize):
    
        self.meshsize = meshsize
        
        super().__init__(
            quadrature_degree = quadrature_degree,
            spatial_order = spatial_order,
            temporal_order = temporal_order)
        
    def init_mesh(self):
        
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def init_integration_measure(self):
        
        self.integration_measure = fe.dx(degree = 4)
        
    def strong_form_residual(self, solution):
        
        mu_s = self.solid_dynamic_viscosity
        
        mu_l = self.liquid_dynamic_viscosity
        
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        t = self.time
        
        grad, dot, div, sym, diff = fe.grad, fe.dot, fe.div, fe.sym, fe.diff
        
        p, u, T = solution
        
        b = self.buoyancy(T)
        
        phil = self.porosity(T)
        
        mu = mu_s + (mu_l - mu_s)*phil
        
        r_p = div(u)
        
        r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(mu*sym(grad(u))) + b
        
        r_T = diff(T + 1./Ste*phil, t) + div(T*u) - 1./Pr*div(grad(T))
        
        return r_p, r_u, r_T
        
    def init_manufactured_solution(self):
        
        pi, sin, cos, exp = fe.pi, fe.sin, fe.cos, fe.exp
        
        x = fe.SpatialCoordinate(self.mesh)
        
        t = self.time
        
        t_f = fe.Constant(1.)
        
        ihat, jhat = self.unit_vectors()
        
        u = exp(t)*sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
            exp(t)*sin(pi*x[0])*sin(2.*pi*x[1])*jhat
        
        p = -sin(pi*x[0])*sin(2.*pi*x[1])
        
        T = 0.5*sin(2.*pi*x[0])*sin(pi*x[1])*(1. - exp(-t**2))
        
        self.manufactured_solution = p, u, T
        
        
def test__verify_spatial_convergence_order_via_mms__second_order(
        constructor_kwargs = {
            "quadrature_degree": 4,
            "spatial_order": 2,
            "temporal_order": 4},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "smoothing": 1./16.},
        mesh_sizes = (5, 10, 20),
        timestep_size = 1./128.,
        tolerance = 0.4):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableModel,
        constructor_kwargs = constructor_kwargs,
        parameters = parameters,
        expected_order = 2,
        mesh_sizes = mesh_sizes,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 0.5,
        plot_solution = False,
        plot_errors = False,
        report = False)
        
        
def test__verify_temporal_convergence_order_via_mms__second_order(
        constructor_kwargs = {
            "quadrature_degree": 4,
            "spatial_order": 2,
            "temporal_order": 2},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "smoothing": 1./16.},
        meshsize = 20,
        timestep_sizes = (1./8., 1./16., 1./32.),
        tolerance = 0.4):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = VerifiableModel,
        constructor_kwargs = constructor_kwargs,
        parameters = parameters,
        expected_order = 2,
        meshsize = meshsize,
        tolerance = tolerance,
        timestep_sizes = timestep_sizes,
        endtime = 0.5,
        plot_solution = False,
        plot_errors = False,
        report = False)
        
        
def test__verify_spatial_convergence_order_via_mms__third_order(
        constructor_kwargs = {
            "quadrature_degree": 8,
            "spatial_order": 3,
            "temporal_order": 3},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "smoothing": 1./16.},
        mesh_sizes = (3, 6, 12),
        timestep_size = 1./128.,
        tolerance = 0.1):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableModel,
        constructor_kwargs = constructor_kwargs,
        parameters = parameters,
        expected_order = 3,
        mesh_sizes = mesh_sizes,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 0.5,
        plot_solution = False,
        plot_errors = False,
        report = False)
        
        
def test__verify_temporal_convergence_order_via_mms__third_order(
        constructor_kwargs = {
            "quadrature_degree": 8,
            "spatial_order": 3,
            "temporal_order": 3},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "smoothing": 1./16.},
        meshsize = 16,
        timestep_sizes = (1./2., 1./4., 1./8., 1./16.),
        tolerance = 0.25):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = VerifiableModel,
        constructor_kwargs = constructor_kwargs,
        parameters = parameters,
        expected_order = 3,
        meshsize = meshsize,
        tolerance = tolerance,
        timestep_sizes = timestep_sizes,
        endtime = 0.5,
        plot_solution = False,
        plot_errors = False,
        report = False)
        