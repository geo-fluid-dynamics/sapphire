import firedrake as fe
import sapphire.simulations.convection_coupled_phasechange
import typing


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
    
    
def heat_driven_cavity_weak_form_residual(sim, solution):
    
    mass = sapphire.simulations.convection_coupled_phasechange.\
        mass(sim, solution)
    
    pressure_penalty = sapphire.simulations.convection_coupled_phasechange.\
        pressure_penalty(sim, solution)
    
    p, u, T = fe.split(solution)
    
    b = water_buoyancy(sim = sim, temperature = T)
    
    Pr = sim.prandtl_number
    
    _, psi_u, psi_T = fe.TestFunctions(sim.function_space)
    
    inner, dot, grad, div, sym = \
        fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
    momentum = dot(psi_u, grad(u)*u + b) \
        - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), sym(grad(u)))
    
    energy = psi_T*dot(u, grad(T)) + dot(grad(psi_T), 1./Pr*grad(T))
    
    return mass + momentum + energy + pressure_penalty
    
    
def dirichlet_boundary_conditions(sim):

    W = sim.function_space
    
    return [fe.DirichletBC(
        W.sub(1), (0.,)*sim.mesh.geometric_dimension(), "on_boundary"),
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
    
    uval = 0.
    
    for unit_vector in sim.unit_vectors():
    
        uval += 0.*unit_vector
        
    u.assign(uval)
    
    T.assign(sim.cold_wall_temperature)
    
    F = heat_driven_cavity_weak_form_residual(
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
                "snes_max_it": sim.solver_parameters["snes_max_it"],
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

    
def weak_form_residual(sim, solution):
    
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
            sapphire.simulations.convection_coupled_phasechange.pressure_penalty)])\
        *fe.dx(degree = sim.quadrature_degree)
    
    
def default_mesh(meshsize, spatial_dimensions):

    if spatial_dimensions == 2:
        
        mesh = fe.UnitSquareMesh(meshsize, meshsize)
        
    elif spatial_dimensions == 3:
    
        mesh = fe.UnitCubeMesh(meshsize, meshsize, meshsize)
    
    return mesh    
    
    
class Simulation(sapphire.simulations.convection_coupled_phasechange.Simulation):

    def __init__(self, *args,
            mesh: typing.Union[fe.UnitSquareMesh, fe.UnitCubeMesh],
            reference_temperature_range__degC = 10.,
            cold_wall_temperature_before_freezing = 0.,
            cold_wall_temperature_during_freezing = -1.,
            stefan_number = 0.125,
            liquidus_temperature = 0.,
            density_solid_to_liquid_ratio = 916.70/999.84,
            heat_capacity_solid_to_liquid_ratio = 0.500,
            thermal_conductivity_solid_to_liquid_ratio = 2.14/0.561,
            **kwargs):
        
        self.reference_temperature_range__degC = fe.Constant(
            reference_temperature_range__degC)
        
        self.hot_wall_temperature = fe.Constant(1.)
        
        self.cold_wall_temperature = fe.Constant(cold_wall_temperature_before_freezing)
            
        super().__init__(
            *args,
            mesh = mesh,
            weak_form_residual = weak_form_residual,
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            stefan_number = stefan_number,
            liquidus_temperature = liquidus_temperature,
            density_solid_to_liquid_ratio = density_solid_to_liquid_ratio,
            heat_capacity_solid_to_liquid_ratio = \
                heat_capacity_solid_to_liquid_ratio,
            thermal_conductivity_solid_to_liquid_ratio = \
                thermal_conductivity_solid_to_liquid_ratio,
            **kwargs)
        
        self.cold_wall_temperature = self.cold_wall_temperature.assign(
            cold_wall_temperature_during_freezing)
        