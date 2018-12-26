import firedrake as fe
import fempy.models.binary_alloy_enthalpy_porosity


class Model(fempy.models.binary_alloy_enthalpy_porosity.Model):

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
        
        b = self.buoyancy(T = T, Cl = 0.)
        
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
            
        if self.solution_file is not None:
            
            self.solution_file.write(
                *self.solution.split(), time = self.time.__float__())
    
        self.cold_wall_temperature.assign(
            self.cold_wall_temperature_during_freezing)
        
        self.initial_values.assign(self.solution)
        
        print("Dropped cold wall temperature")
        
        print("Running solidification")
        
        super().run(endtime = endtime, plot = plot)
        