import firedrake as fe
import sapphire.simulations.convection_coupled_alloy_phasechange


initial_porosity = 1.

basesim_module = sapphire.simulations.convection_coupled_alloy_phasechange

BaseSim = basesim_module.Simulation

def nsb_element(cell, degree):
    """ Equal-order mixed finite element for pressure, velocity, temperature, solute"""
    scalar = fe.FiniteElement("P", cell, degree)
    
    vector = fe.VectorElement("P", cell, degree)
    
    return fe.MixedElement(scalar, vector, scalar)
    
def nsb_variational_form_residual(sim, solution):
    """ For initial values, 
    solve steady state Navier-Stokes-Boussinesq for all liquid, no solute 
    """
    p, u, h = fe.split(solution)
    
    gamma = sim.pressure_penalty_factor
    
    T = basesim_module.temperature(
        sim = sim, enthalpy = h, porosity = initial_porosity)
    
    b = basesim_module.buoyancy(
        sim = sim, temperature = T, liquid_solute_concentration = 0.)
    
    Pr = sim.prandtl_number
    
    ghat = fe.Constant(-sapphire.simulation.unit_vectors(sim.mesh)[1])
    
    psi_p, psi_u, psi_h = fe.TestFunctions(solution.function_space())
    
    inner, dot, grad, div, sym = \
        fe.inner, fe.dot, fe.grad, fe.div, fe.sym
    
    mass = psi_p*div(u)
    
    momentum = dot(psi_u, grad(u)*u + Pr*b*ghat) \
        - div(psi_u)*p + Pr*inner(sym(grad(psi_u)), sym(grad(u)))
    
    energy = psi_h*dot(u, grad(T)) + dot(grad(psi_h), grad(T))
    
    stabilization = gamma*psi_p*p
    
    return mass + momentum + energy + stabilization
    
def initial_dirichlet_boundary_conditions(sim):

    W = sim.function_space
    
    T_h = sim.max_temperature
    
    h_h = basesim_module.enthalpy(
        sim = sim,
        temperature = T_h,
        porosity = initial_porosity)
    
    T_c0 = sim.initial_cold_wall_temperature
    
    h_c0 = basesim_module.enthalpy(
        sim = sim,
        temperature = T_c0,
        porosity = initial_porosity)
    
    return [fe.DirichletBC(
        W.sub(1), (0., 0.), 2),
        fe.DirichletBC(W.sub(2), h_h, 1),
        fe.DirichletBC(W.sub(2), h_c0, 2)]
    
def initial_values(sim):
    
    print("Solving steady heat driven cavity to obtain initial values")
    
    mesh = sim.mesh
    
    element = nsb_element(cell = mesh.ufl_cell(), degree = sim.element_degree)
    
    W_nsb = fe.FunctionSpace(mesh, element)
    
    w_nsb = fe.Function(W_nsb)
    
    p, u, h = w_nsb.split()
    
    p.assign(0.)
    
    ihat, jhat = sim.unit_vectors()
    
    u.assign(0.*ihat + 0.*jhat)
    
    T_c0 = sim.initial_cold_wall_temperature
    
    h_c0 = basesim_module.enthalpy(
        sim = sim,
        temperature = T_c0,
        porosity = initial_porosity)
    
    h.assign(h_c0)
    
    F = nsb_variational_form_residual(
        sim = sim,
        solution = w_nsb)*fe.dx(degree = sim.quadrature_degree)
        
    problem = fe.NonlinearVariationalProblem(
        F = F,
        u = w_nsb,
        bcs = initial_dirichlet_boundary_conditions(sim),
        J = fe.derivative(F, w_nsb))
    
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
        
        return w_nsb
    
    w_nsb, _ = \
        sapphire.continuation.solve_with_bounded_regularization_sequence(
            solve = solve,
            solution = w_nsb,
            backup_solution = fe.Function(w_nsb),
            regularization_parameter = sim.thermal_rayleigh_number,
            initial_regularization_sequence = (
                0., sim.thermal_rayleigh_number.__float__()))
    
    w_0 = fe.Function(sim.function_space)
    
    for w_0_i, w_nsb_i in zip(w_0.split()[:-1], w_nsb.split()):

        w_0_i.assign(w_nsb_i)
    
    _, _, _, S_l = w_0.split()
    
    S_l = S_l.assign(sim.initial_solute_concentration)
    
    return w_0

    
def dirichlet_boundary_conditions(sim):

    W = sim.function_space
    
    h_h = basesim_module.enthalpy(
        sim = sim,
        temperature = sim.max_temperature,
        porosity = initial_porosity)
    
    h_c = basesim_module.enthalpy(
        sim = sim,
        temperature = sim.cold_wall_temperature,
        porosity = sim.cold_wall_porosity)
    
    return [fe.DirichletBC(
        W.sub(1), (0., 0.), 2),
        fe.DirichletBC(W.sub(2), h_h, 1),
        fe.DirichletBC(W.sub(2), h_c, 2)]
        
        
class Simulation(BaseSim):

    def __init__(self, *args, 
            meshsize, 
            depth = 2.,
            initial_cold_wall_temperature,
            cold_wall_temperature,
            initial_solute_concentration,
            cold_wall_porosity,
            **kwargs):
        
        self.initial_cold_wall_temperature = fe.Constant(
            initial_cold_wall_temperature)
        
        self.cold_wall_temperature = fe.Constant(cold_wall_temperature)
        
        self.initial_solute_concentration = fe.Constant(initial_solute_concentration)
        
        self.cold_wall_porosity = fe.Constant(cold_wall_porosity)
        
        super().__init__(
            *args,
            mesh = fe.PeriodicRectangleMesh(meshsize, meshsize, 1., depth, direction = "x"),
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        