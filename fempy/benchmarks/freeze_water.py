import firedrake as fe
import fempy.models.enthalpy_porosity
import numpy


class Model(fempy.models.enthalpy_porosity.Model):

    def __init__(self, *args, meshsize, **kwargs):
        
        self.meshsize = meshsize
        
        self.reference_temperature_range__degC = fe.Constant(10.)  # [deg C]
        
        self.hot_wall_temperature = fe.Constant(1.)
    
        self.cold_wall_temperature_before_freezing = fe.Constant(0.)
        
        self.cold_wall_temperature_during_freezing = fe.Constant(-1.)
        
        self.cold_wall_temperature = fe.Constant(0.)
        
        self.cold_wall_temperature.assign(
            self.cold_wall_temperature_before_freezing)
        
        super().__init__(*args, **kwargs)
        
        Ra = 2.518084e6
        
        Pr = 6.99
        
        self.grashof_number.assign(Ra/Pr)
        
        self.prandtl_number.assign(Pr)
        
        self.stefan_number.assign(0.125)
        
        self.liquidus_temperature.assign(0.)
        
        self.heat_capacity_solid_to_liquid_ratio.assign(0.500)
        
        self.thermal_conductivity_solid_to_liquid_ratio.assign(2.14/0.561)

    def buoyancy(self, T):
        """ Eq. (25) from @cite{danaila2014newton} """
        T_anomaly_degC = fe.Constant(4.0293)  # [deg C]
        
        rho_anomaly_SI = fe.Constant(999.972)  # [kg/m^3]
        
        w = fe.Constant(9.2793e-6)  # [(deg C)^(-q)]
        
        q = fe.Constant(1.894816)
        
        M = self.reference_temperature_range__degC
        
        T_L = self.liquidus_temperature
        
        def T_degC(T):
            """ T = (T_degC - T_L)/M """
            return M*T + T_L
        
        def rho_of_T_degC(T_degC):
            """ Eq. (24) from @cite{danaila2014newton} """
            return rho_anomaly_SI*(1. - w*abs(T_degC - T_anomaly_degC)**q)
            
        def rho(T):
            
            return rho_of_T_degC(T_degC(T))
        
        beta = fe.Constant(6.91e-5)  # [K^-1]
        
        Gr = self.grashof_number
        
        _, _, T = fe.split(self.solution)
        
        _, jhat = self.unit_vectors()
        
        ghat = fe.Constant(-jhat)
        
        return Gr/(beta*M)*(rho(T_L) - rho(T))/rho(T_L)*ghat
        
    def init_mesh(self):
    
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def initial_values(self):
        
        print("Solving steady heat driven cavity to obtain initial values")
        
        r = self.heat_driven_cavity_weak_form_residual()\
            *self.integration_measure
        
        u = self.solution
        
        problem = fe.NonlinearVariationalProblem(
            r, u, self.dirichlet_boundary_conditions, fe.derivative(r, u))
        
        solver = fe.NonlinearVariationalSolver(
            problem, 
            solver_parameters = {
                "snes_type": "newtonls",
                "snes_max_it": 50,
                "snes_monitor": True,
                "ksp_type": "preonly", 
                "pc_type": "lu", 
                "mat_type": "aij",
                "pc_factor_mat_solver_type": "mumps"})
        
        x = fe.SpatialCoordinate(self.mesh)
        
        T_c = self.cold_wall_temperature_before_freezing.__float__()
        
        u.assign(fe.interpolate(
            fe.Expression(
                (0., 0., 0., T_c),
                element = self.element),
            self.function_space))
            
        self.cold_wall_temperature.assign(
            self.cold_wall_temperature_before_freezing)
        
        T_h = self.hot_wall_temperature.__float__()
        
        fempy.continuation.solve(
            model = self,
            solver = solver,
            continuation_parameter = self.grashof_number,
            continuation_sequence = None,
            leftval = 0.,
            rightval = self.grashof_number.__float__(),
            startleft = True,
            maxcount = 16)
        
        self.cold_wall_temperature.assign(
            self.cold_wall_temperature_during_freezing)
        
        return self.solution
        
    def init_dirichlet_boundary_conditions(self):
    
        W = self.function_space
        
        self.dirichlet_boundary_conditions = [
            fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
            fe.DirichletBC(W.sub(2), self.hot_wall_temperature, 1),
            fe.DirichletBC(W.sub(2), self.cold_wall_temperature, 2)]
            
    def heat_driven_cavity_weak_form_residual(self):
        
        mass = self.mass()
        
        stabilization = self.stabilization()
        
        p, u, T = fe.split(self.solution)
        
        b = self.buoyancy(T)
        
        Pr = self.prandtl_number
        
        _, psi_u, psi_T = fe.TestFunctions(self.function_space)
        
        inner, dot, grad, div, sym = \
            fe.inner, fe.dot, fe.grad, fe.div, fe.sym
            
        momentum = dot(psi_u, grad(u)*u + b) \
            - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), sym(grad(u)))
        
        energy = psi_T*dot(u, grad(T)) + dot(grad(psi_T), 1./Pr*grad(T))
        
        return mass + momentum + energy + stabilization
        