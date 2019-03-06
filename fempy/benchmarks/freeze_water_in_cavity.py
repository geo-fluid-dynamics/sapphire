import firedrake as fe
import fempy.models.convection_coupled_phasechange


def water_buoyancy(model, temperature):
    """ Eq. (25) from @cite{danaila2014newton} """
    T = temperature
    
    T_anomaly_degC = fe.Constant(4.0293)
    
    rho_anomaly_SI = fe.Constant(999.972)
    
    w_degC = fe.Constant(9.2793e-6)
    
    q = fe.Constant(1.894816)
    
    M = model.reference_temperature_range__degC
    
    T_L = model.liquidus_temperature
    
    def T_degC(T):
        """ T = (T_degC - T_L)/M """
        return M*T + T_L
    
    def rho_of_T_degC(T_degC):
        """ Eq. (24) from @cite{danaila2014newton} """
        return rho_anomaly_SI*(1. - w_degC*abs(T_degC - T_anomaly_degC)**q)
        
    def rho(T):
        
        return rho_of_T_degC(T_degC(T))
    
    beta = fe.Constant(6.91e-5)  # [K^-1]
    
    Gr = model.grashof_number
    
    ghat = fe.Constant(-fempy.model.unit_vectors(model.mesh)[1])
    
    return Gr/(beta*M)*(rho(T_L) - rho(T))/rho(T_L)*ghat
    
    
def heat_driven_cavity_variational_form_residual(model, solution):
    
    mass = fempy.models.convection_coupled_phasechange.mass(model, solution)
    
    stabilization = fempy.models.convection_coupled_phasechange.stabilization(
        model, solution)
    
    p, u, T = fe.split(solution)
    
    b = water_buoyancy(model = model, temperature = T)
    
    Pr = model.prandtl_number
    
    _, psi_u, psi_T = fe.TestFunctions(model.function_space)
    
    inner, dot, grad, div, sym = \
        fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
    momentum = dot(psi_u, grad(u)*u + b) \
        - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), sym(grad(u)))
    
    energy = psi_T*dot(u, grad(T)) + dot(grad(psi_T), 1./Pr*grad(T))
    
    return mass + momentum + energy + stabilization
    
    
def dirichlet_boundary_conditions(model):

    W = model.function_space
    
    dim = model.mesh.geometric_dimension()
    
    return [fe.DirichletBC(
        W.sub(1), (0.,)*dim, "on_boundary"),
        fe.DirichletBC(W.sub(2), model.hot_wall_temperature, 1),
        fe.DirichletBC(W.sub(2), model.cold_wall_temperature, 2)]
        
    
def initial_values(model):
    
    print("Solving steady heat driven cavity to obtain initial values")
    
    Ra = 2.518084e6

    Pr = 6.99

    model.grashof_number = model.grashof_number.assign(Ra/Pr)
    
    model.prandtl_number = model.prandtl_number.assign(Pr)
    
    dim = model.mesh.geometric_dimension()
    
    T_c = model.cold_wall_temperature.__float__()
    
    w = fe.interpolate(
        fe.Expression(
            (0.,) + (0.,)*dim + (T_c,),
            element = model.element),
        model.function_space)
    
    r = heat_driven_cavity_variational_form_residual(
        model = model, solution = w)*fe.dx(degree = model.quadrature_degree)
        
    T_h = model.hot_wall_temperature.__float__()
    
    w, _, _ = \
        fempy.continuation.solve_with_auto_continuation(
            solve = lambda: fempy.model.solve(
                variational_form_residual = r,
                solution = w,
                dirichlet_boundary_conditions = \
                    dirichlet_boundary_conditions(model)),
            solution = w,
            continuation_parameter = model.grashof_number,
            continuation_sequence = (0., model.grashof_number.__float__()),
            startleft = True,
            maxcount = 16)
    
    return w

    
def variational_form_residual(model, solution):
    
    return sum(
    [r(model = model, solution = solution)
        for r in (
            fempy.models.convection_coupled_phasechange.mass,
            lambda model, solution: \
                fempy.models.convection_coupled_phasechange.momentum(
                    model = model,
                    solution = solution,
                    buoyancy = water_buoyancy),
            fempy.models.convection_coupled_phasechange.energy,
            fempy.models.convection_coupled_phasechange.stabilization)])\
        *fe.dx(degree = model.quadrature_degree)
    
    
class Model(fempy.models.convection_coupled_phasechange.Model):

    def __init__(self, *args, spatial_dimensions, meshsize, **kwargs):
        
        self.reference_temperature_range__degC = fe.Constant(10.)
        
        self.hot_wall_temperature = fe.Constant(1.)
        
        self.cold_wall_temperature = fe.Constant(0.)
        
        if spatial_dimensions == 2:
    
            Mesh = fe.UnitSquareMesh
            
        elif spatial_dimensions == 3:
        
            Mesh = fe.UnitCubeMesh
        
        super().__init__(
            *args,
            mesh = Mesh(*(meshsize,)*spatial_dimensions),
            variational_form_residual = variational_form_residual,
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        
        self.stefan_number = self.stefan_number.assign(0.125)
        
        self.liquidus_temperature = self.liquidus_temperature.assign(0.)
        
        self.heat_capacity_solid_to_liquid_ratio = \
            self.heat_capacity_solid_to_liquid_ratio.assign(0.500)
        
        self.thermal_conductivity_solid_to_liquid_ratio = \
            self.thermal_conductivity_solid_to_liquid_ratio.assign(2.14/0.561)
        
        self.cold_wall_temperature = self.cold_wall_temperature.assign(-1.)
        
    def run(self, *args, endtime, **kwargs):
        
        for i in range(2):
        
            self.solutions, self.time, self.snes_iteration_count = \
                super().run(*args,
                    endtime = self.time.__float__() + self.timestep_size.__float__(),
                    new_smoothing_sequence = (4., self.smoothing.__float__()),
                    **kwargs)
        
        return super().run(*args,
            endtime = endtime,
            new_smoothing_sequence = False,
            **kwargs)
        