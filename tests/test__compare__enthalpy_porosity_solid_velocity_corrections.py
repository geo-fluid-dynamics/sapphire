import firedrake as fe 
import fempy.mms
import fempy.models.enthalpy_porosity
import fempy.benchmarks.melt_octadecane


class GeneralSolidVelocityCorrectionsModel(fempy.models.enthalpy_porosity.Model):
    
    def __init__(self, *args, **kwargs):
    
        self.liquid_dynamic_viscosity = fe.Constant(1.)
        
        self.solid_dynamic_viscosity = fe.Constant(1.)
        
        self.liquidus_temperature_offset = fe.Constant(0.)
        
        super().__init__(*args, **kwargs)
        
    def porosity(self, T):
        """ Regularization from @cite{zimmerman2018monolithic} """
        T_L = self.liquidus_temperature
        
        delta_T_L = self.liquidus_temperature_offset
        
        s = self.smoothing
        
        tanh = fe.tanh
        
        return 0.5*(1. + tanh((T - (T_L + delta_T_L))/s))
        
    def dynamic_viscosity(self, T):
        
        mu_s = self.solid_dynamic_viscosity
        
        mu_l = self.liquid_dynamic_viscosity
        
        phil = self.porosity(T)
        
        return mu_s + (mu_l - mu_s)*phil
        
    def momentum(self):
        
        p, u, T = fe.split(self.solution)
        
        _, u_t, _, _, _ = self.time_discrete_terms
        
        b = self.buoyancy(T)
        
        mu = self.dynamic_viscosity(T)
        
        d = self.solid_velocity_relaxation(T)
        
        _, psi_u, _ = fe.TestFunctions(self.function_space)
        
        inner, dot, grad, div, sym = \
            fe.inner, fe.dot, fe.grad, fe.div, fe.sym
            
        return dot(psi_u, u_t + grad(u)*u + b + d*u) \
            - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), mu*sym(grad(u)))

            
class VerifiableGeneralSolidVelocityCorrectionsModel(
        GeneralSolidVelocityCorrectionsModel):

    def __init__(self, *args, meshsize, **kwargs):
    
        self.meshsize = meshsize
        
        super().__init__(*args, **kwargs)
        
    def init_mesh(self):
        
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def strong_form_residual(self, solution):
        
        mu_s = self.solid_dynamic_viscosity
        
        mu_l = self.liquid_dynamic_viscosity
        
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        t = self.time
        
        grad, dot, div, sym, diff = fe.grad, fe.dot, fe.div, fe.sym, fe.diff
        
        p, u, T = solution
        
        b = self.buoyancy(T)
        
        d = self.solid_velocity_relaxation(T)
        
        phil = self.porosity(T)
        
        mu = mu_s + (mu_l - mu_s)*phil
        
        r_p = div(u)
        
        r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(mu*sym(grad(u))) \
            + b + d*u
        
        r_T = diff(T + 1./Ste*phil, t) + dot(u, grad(T)) - 1./Pr*div(grad(T))
        
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
        
        
class VerifiableSolidViscosityModel(
        VerifiableGeneralSolidVelocityCorrectionsModel):
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.solid_dynamic_viscosity.assign(1.e8)
        
        self.solid_velocity_relaxation_factor.assign(1.e32)

        
def test__verify__solid_viscosity__second_order_spatial_convergence__via_mms(
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
        tolerance = 0.06):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableSolidViscosityModel,
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
        
        
def test__verify__solid_viscosity__second_order_temporal_convergence__via_mms(
        constructor_kwargs = {
            "quadrature_degree": 4,
            "spatial_order": 4,
            "temporal_order": 2},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "smoothing": 1./16.},
        meshsize = 20,
        timestep_sizes = (1./4., 1./8., 1./16., 1./32.),
        tolerance = 0.4):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = VerifiableSolidViscosityModel,
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
        
        
class VerifiableKozenyCarmanModel(
        VerifiableGeneralSolidVelocityCorrectionsModel):

    def __init__(self, *args, **kwargs):
        
        self.small_number_to_avoid_division_by_zero = fe.Constant(1.e-8)
        
        super().__init__(*args, **kwargs)
        
    def solid_velocity_relaxation(self, T):
        """ Kozeny-Carman relation """
        tau = self.solid_velocity_relaxation_factor
        
        epsilon = self.small_number_to_avoid_division_by_zero
        
        phil = self.porosity(T)
        
        return 1./tau*(1. - phil)**2/(phil**3 + epsilon)
        
    def dynamic_viscosity(self, T):
        
        return self.liquid_dynamic_viscosity

        
def test__verify__kozeny_carman__second_order_spatial_convergence__via_mms(
        constructor_kwargs = {
            "quadrature_degree": 4,
            "spatial_order": 2,
            "temporal_order": 4},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "smoothing": 1./16.},
        mesh_sizes = (2, 4, 8, 16, 32),
        timestep_size = 1./128.,
        tolerance = 0.4):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableKozenyCarmanModel,
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
        
        
def test__verify__kozeny_carman__second_order_temporal_convergence__via_mms(
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
        tolerance = 0.02):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = VerifiableKozenyCarmanModel,
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
        
        
class GeneralSolidVelocityCorrectionsMeltingOctadecaneModel(
        fempy.benchmarks.melt_octadecane.Model):
    
    def __init__(self, *args, **kwargs):
    
        self.liquid_dynamic_viscosity = fe.Constant(1.)
        
        self.solid_dynamic_viscosity = fe.Constant(1.)
        
        self.liquidus_temperature_offset = fe.Constant(0.)
        
        super().__init__(*args, **kwargs)
        
    def porosity(self, T):
        """ Regularization from @cite{zimmerman2018monolithic} """
        T_L = self.liquidus_temperature
        
        delta_T_L = self.liquidus_temperature_offset
        
        s = self.smoothing
        
        tanh = fe.tanh
        
        return 0.5*(1. + tanh((T - (T_L + delta_T_L))/s))
        
    def dynamic_viscosity(self, T):
        
        mu_s = self.solid_dynamic_viscosity
        
        mu_l = self.liquid_dynamic_viscosity
        
        phil = self.porosity(T)
        
        return mu_s + (mu_l - mu_s)*phil
        
    def momentum(self):
        
        p, u, T = fe.split(self.solution)
        
        _, u_t, _, _, _ = self.time_discrete_terms
        
        b = self.buoyancy(T)
        
        mu = self.dynamic_viscosity(T)
        
        d = self.solid_velocity_relaxation(T)
        
        _, psi_u, _ = fe.TestFunctions(self.function_space)
        
        inner, dot, grad, div, sym = \
            fe.inner, fe.dot, fe.grad, fe.div, fe.sym
            
        return dot(psi_u, u_t + grad(u)*u + b + d*u) \
            - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), mu*sym(grad(u)))
        
        
class SolidViscosityMeltingOctadecaneModel(
        GeneralSolidVelocityCorrectionsMeltingOctadecaneModel):
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.solid_dynamic_viscosity.assign(1.e8)
        
        self.solid_velocity_relaxation_factor.assign(1.e32)
        
        
def test__regression__validate__solid_viscosity__melt_octadecane():
    
    endtime, expected_liquid_area, tolerance = 30., 0.21, 0.01
    
    nx = 32
    
    Delta_t = 10.
    
    model = SolidViscosityMeltingOctadecaneModel(
        quadrature_degree = 4, spatial_order = 2, temporal_order = 2, meshsize = nx)
    
    model.timestep_size.assign(Delta_t)
    
    s = 1./256.
    
    model.smoothing.assign(s)
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "melt_octadecane/solid_viscosity/second_order/" 
        + "nx" + str(nx) + "_Deltat" + str(Delta_t) 
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
    