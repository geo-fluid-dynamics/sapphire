import firedrake as fe
import sapphire.simulations.convection_coupled_phasechange


def water_buoyancy(sim, temperature):
    """ Eq. (25) from @cite{danaila2014newton} """
    T = temperature
    
    T_anomaly_degC = fe.Constant(4.0293)
    
    rho_anomaly_SI = fe.Constant(999.972)
    
    w_degC = fe.Constant(9.2793e-6)
    
    q = fe.Constant(1.894816)
    
    M = sim.reference_temperature_range__degC
    
    def T_degC(T):
        """ T = T_degC/M """
        return M*T
    
    def rho_of_T_degC(T_degC):
        """ Eq. (24) from @cite{danaila2014newton} """
        return rho_anomaly_SI*(1. - w_degC*abs(T_degC - T_anomaly_degC)**q)
        
    def rho(T):
        
        return rho_of_T_degC(T_degC(T))
    
    beta = fe.Constant(6.91e-5)  # [K^-1]
    
    Gr = sim.grashof_number
    
    ghat = fe.Constant(-sapphire.simulation.unit_vectors(sim.mesh)[1])
    
    rho_0 = rho(T = 0.)
    
    return Gr/(beta*M)*(rho_0 - rho(T))/rho_0*ghat
    
    
def heat_driven_cavity_variational_form_residual(sim, solution):
    
    mass = sapphire.simulations.convection_coupled_phasechange.\
        mass(sim, solution)
    
    stabilization = sapphire.simulations.convection_coupled_phasechange.\
        stabilization(sim, solution)
    
    p, u, T = fe.split(solution)
    
    b = water_buoyancy(sim = sim, temperature = T)
    
    Pr = sim.prandtl_number
    
    _, psi_u, psi_T = fe.TestFunctions(sim.function_space)
    
    inner, dot, grad, div, sym = \
        fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
    momentum = dot(psi_u, grad(u)*u + b) \
        - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), sym(grad(u)))
    
    energy = psi_T*dot(u, grad(T)) + dot(grad(psi_T), 1./Pr*grad(T))
    
    return mass + momentum + energy + stabilization
    
    
def dirichlet_boundary_conditions(sim):

    W = sim.function_space
    
    return [fe.DirichletBC(
        W.sub(1), (0., 0.), "on_boundary"),
        fe.DirichletBC(W.sub(2), sim.hot_wall_temperature, 1),
        fe.DirichletBC(W.sub(2), sim.cold_wall_temperature, 2)]
        
    
def initial_values(sim):
    
    print("Solving steady heat driven cavity to obtain initial values")
    
    Ra = 2.518084e6

    Pr = 6.99

    sim.grashof_number = sim.grashof_number.assign(Ra/Pr)
    
    sim.prandtl_number = sim.prandtl_number.assign(Pr)
    
    w = fe.Function(sim.function_space)
    
    p, u, T = w.split()
    
    p.assign(0.)
    
    ihat, jhat = sim.unit_vectors()
    
    u.assign(0.*ihat + 0.*jhat)
    
    T.assign(sim.cold_wall_temperature)
    
    F = heat_driven_cavity_variational_form_residual(
        sim = sim,
        solution = w)*fe.dx(degree = sim.quadrature_degree)
        
    problem = fe.NonlinearVariationalProblem(
        F = F,
        u = w,
        bcs = dirichlet_boundary_conditions(sim),
        J = fe.derivative(F, w))
    
    solver = fe.NonlinearVariationalSolver(
        problem = problem,
        solver_parameters = {
                "snes_type": "newtonls",
                "snes_monitor": None,
                "ksp_type": "preonly", 
                "pc_type": "lu", 
                "mat_type": "aij",
                "pc_factor_mat_solver_type": "mumps"})
                
    def solve():
    
        solver.solve()
        
        return w
    
    w, _ = \
        sapphire.continuation.solve_with_bounded_regularization_sequence(
            solve = solve,
            solution = w,
            backup_solution = fe.Function(w),
            regularization_parameter = sim.grashof_number,
            initial_regularization_sequence = (
                0., sim.grashof_number.__float__()))
                
    return w

    
def variational_form_residual(sim, solution):
    
    return sum(
    [r(sim = sim, solution = solution)
        for r in (
            sapphire.simulations.convection_coupled_phasechange.mass,
            lambda sim, solution: \
                sapphire.simulations.convection_coupled_phasechange.momentum(
                    sim = sim,
                    solution = solution,
                    buoyancy = water_buoyancy),
            sapphire.simulations.convection_coupled_phasechange.energy,
            sapphire.simulations.convection_coupled_phasechange.stabilization)])\
        *fe.dx(degree = sim.quadrature_degree)
    
    
class Simulation(sapphire.simulations.convection_coupled_phasechange.Simulation):

    def __init__(self, *args, meshsize, **kwargs):
        
        self.reference_temperature_range__degC = fe.Constant(10.)
        
        self.hot_wall_temperature = fe.Constant(1.)
        
        self.cold_wall_temperature = fe.Constant(0.)
        
        super().__init__(
            *args,
            mesh = fe.UnitSquareMesh(meshsize, meshsize),
            variational_form_residual = variational_form_residual,
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        
        self.stefan_number = self.stefan_number.assign(0.125)
        
        self.liquidus_temperature = self.liquidus_temperature.assign(0.)
        
        self.density_solid_to_liquid_ratio = \
            self.density_solid_to_liquid_ratio.assign(916.70/999.84)
        
        self.heat_capacity_solid_to_liquid_ratio = \
            self.heat_capacity_solid_to_liquid_ratio.assign(0.500)
        
        self.thermal_conductivity_solid_to_liquid_ratio = \
            self.thermal_conductivity_solid_to_liquid_ratio.assign(2.14/0.561)
        
        self.cold_wall_temperature = self.cold_wall_temperature.assign(-1.)
        