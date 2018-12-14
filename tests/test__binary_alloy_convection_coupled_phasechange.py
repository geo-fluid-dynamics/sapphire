import firedrake as fe
import fempy.models.binary_alloy_convection_coupled_phasechange
import pathlib


class Model(fempy.models.binary_alloy_convection_coupled_phasechange.Model):

    def __init__(self, meshsize):
        
        self.meshsize = meshsize
        
        self.hot_wall_temperature = fe.Constant(1.)
    
        self.cold_wall_temperature_before_freezing = fe.Constant(0.01)
        
        self.cold_wall_temperature_during_freezing = fe.Constant(-1.)
        
        self.initial_concentration = fe.Constant(1.)
        
        self.cold_wall_temperature = fe.Constant(0.)
        
        self.cold_wall_temperature.assign(
            self.cold_wall_temperature_before_freezing)
        
        super().__init__()
        
        delattr(self, "element")
        
        self.init_heat_driven_cavity_weak_form_residual()
        
        self.init_heat_driven_cavity_problem()
        
        self.init_heat_driven_cavity_solver()

    def init_mesh(self):
    
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def init_function_space(self):
    
        P1 = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        
        P2 = fe.VectorElement("P", self.mesh.ufl_cell(), 2)
        
        self.navier_stokes_boussinesq_function_space = fe.FunctionSpace(
            self.mesh, fe.MixedElement(P1, P2, P1))
        
        self.concentration_function_space = fe.FunctionSpace(self.mesh, P1)
        
        self.function_space = self.navier_stokes_boussinesq_function_space*\
            self.concentration_function_space
        
    def init_initial_values(self):
        
        self.initial_values = fe.Function(self.function_space)
        
    def init_dirichlet_boundary_conditions(self):
    
        W = self.function_space
        
        self.dirichlet_boundary_conditions = [
            fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
            fe.DirichletBC(W.sub(2), self.hot_wall_temperature, 1),
            fe.DirichletBC(W.sub(2), self.cold_wall_temperature, 2)]
           
    def init_solution(self):
    
        super().init_solution()
        
        self.heat_driven_cavity_solution = fe.Function(
            self.navier_stokes_boussinesq_function_space)
           
    def init_heat_driven_cavity_weak_form_residual(self):
        """ Weak form from @cite{zimmerman2018monolithic} """
        mu = self.liquid_dynamic_viscosity
        
        Pr = self.prandtl_number
        
        inner, dot, grad, div, sym = \
            fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
        p, u, T = fe.split(self.heat_driven_cavity_solution)
        
        b = self.buoyancy(T = T, C = 0.)
        
        psi_p, psi_u, psi_T = fe.TestFunctions(
            self.navier_stokes_boussinesq_function_space)
        
        mass = psi_p*div(u)
        
        momentum = dot(psi_u, grad(u)*u + b) \
            - div(psi_u)*p + 2.*mu*inner(sym(grad(psi_u)), sym(grad(u)))
        
        energy = dot(grad(psi_T), 1./Pr*grad(T) - T*u)
        
        gamma = self.pressure_penalty_factor
        
        stabilization = gamma*psi_p*p
        
        self.heat_driven_cavity_weak_form_residual = mass + momentum + energy \
            + stabilization
            
    def init_heat_driven_cavity_problem(self):
    
        dx = self.integration_measure
        
        r = self.heat_driven_cavity_weak_form_residual*dx
        
        u = self.heat_driven_cavity_solution
        
        self.heat_driven_cavity_problem = fe.NonlinearVariationalProblem(
            r, u, self.dirichlet_boundary_conditions, fe.derivative(r, u))
            
    def init_heat_driven_cavity_solver(self, solver_parameters = {
                "snes_monitor": True,
                "ksp_type": "preonly",
                "mat_type": "aij",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"}):
        
        self.heat_driven_cavity_solver = fe.NonlinearVariationalSolver(
            self.heat_driven_cavity_problem, 
            solver_parameters = solver_parameters)
            
    def run(self, endtime, plot = False):
        
        self.cold_wall_temperature.assign(
            self.cold_wall_temperature_before_freezing)
        
        print("Running steady state heat-driven cavity")
        
        self.heat_driven_cavity_solver.solve()
        
        for u, u_ in zip(
                self.solution.split()[:-1],
                self.heat_driven_cavity_solution.split()):
        
            u.assign(u_)
            
        self.solution.split()[-1].assign(fe.interpolate(
            self.initial_concentration, self.concentration_function_space))
        
        if plot:
            
            self.plot()
            
        self.cold_wall_temperature.assign(
            self.cold_wall_temperature_during_freezing)
        
        self.initial_values.assign(self.solution)
        
        print("Dropped cold wall temperature")
        
        print("Running solidification")
        
        super().run(endtime = endtime, plot = plot)


def test__convection_coupled_sea_ice_cavity_freezing__regression():

    endtime = 1.
    
    timestep_size = 1./32.
    
    meshsize = 32
    
    phase_interface_smoothing = 1./64.
    
    expected_solid_area = 0.080
    
    tolerance = 0.001
    
    
    model = Model(meshsize = meshsize)
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "convection_coupled_sea_ice_cavity_freezing/")
    
    model.hot_wall_temperature.assign(1./3.)
    
    model.cold_wall_temperature_before_freezing.assign(0.)
    
    model.cold_wall_temperature_during_freezing.assign(-2./3.)
    
    model.temperature_rayleigh_number.assign(1.e6)
    
    model.concentration_rayleigh_number.assign(-1.e6)
    
    model.prandtl_number.assign(13.)
    
    model.schmidt_number.assign(1100.)
    
    model.stefan_number.assign(0.19)
    
    model.pure_liquidus_temperature.assign(0.)
    
    model.initial_concentration.assign(1.)
    
    model.liquidus_slope.assign(-0.1)
    
    model.phase_interface_smoothing.assign(phase_interface_smoothing)
    
    model.timestep_size.assign(timestep_size)
    
    
    model.run(endtime = endtime, plot = False)
    
    
    p, u, T, C = model.solution.split()
    
    phi = model.semi_phasefield(T = T, C = C)
    
    A_S = fe.assemble(phi*fe.dx)
    
    print("Solid area = " + str(A_S))
    
    assert(abs(A_S - expected_solid_area) < tolerance)
        
    
class VerifiableModel(
    fempy.models.binary_alloy_convection_coupled_phasechange.Model):

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
        
        self.phase_interface_smoothing.assign(1./32.)
        
        self.smoothing_sequence = None
        
    def init_mesh(self):
        
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def strong_form_residual(self, solution):
        
        gamma = self.pressure_penalty_factor
        
        mu_S = self.solid_dynamic_viscosity
        
        mu_L = self.liquid_dynamic_viscosity
        
        Ra = self.rayleigh_number
        
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        Sc = self.schmidt_number
        
        t = self.time
        
        grad, dot, div, sym, diff = fe.grad, fe.dot, fe.div, fe.sym, fe.diff
        
        p, u, T, C = solution
        
        b = self.buoyancy(T, C)
        
        phi = self.semi_phasefield(T, C)
        
        mu = mu_L + (mu_S - mu_L)*phi
        
        r_p = div(u) + gamma*p
        
        r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(mu*sym(grad(u))) + b
        
        r_T = diff(T, t) - 1./Ste*diff(phi, t) + div(T*u) - 1./Pr*div(grad(T))
        
        r_C = (1. - phi)*diff(C, t) + div(C*u) - 1./Sc*div((1. - phi)*grad(C)) \
            - C*diff(phi, t)
        
        return r_p, r_u, r_T, r_C
        
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
        
        C = sin(pi*x[0])*sin(2.*pi*x[1])*exp(-2.*t**2)
        
        self.manufactured_solution = p, u, T, C
        
    def init_initial_values(self):
        
        self.initial_values = fe.Function(self.function_space)
        
        for u_m, V in zip(
                self.manufactured_solution, self.function_space):
        
            self.initial_values.assign(fe.interpolate(u_m, V))
        
    def solve(self):
        
        self.solver.parameters["snes_monitor"] = False
        
        super().solve()
            
        print("Solved at time t = " + str(self.time.__float__()))
        
        
def test__verify_spatial_convergence_order_via_mms(
        parameters = {
            "temperature_rayleigh_number": 8.,
            "concentration_rayleigh_number": -9.,
            "prandtl_number": 7.,
            "stefan_number": 0.2,
            "schmidt_number": 6.,
            "pure_liquidus_temperature": 0.,
            "liquidus_slope": -0.11,
            "phase_interface_smoothing": 1./16.,
            "autosmooth_maxcount": 16},
        grid_sizes = (4, 8, 16),
        timestep_size = 1./64.,
        tolerance = 0.1):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableModel,
        parameters = parameters,
        expected_order = 2,
        grid_sizes = grid_sizes,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 1.)
    
    