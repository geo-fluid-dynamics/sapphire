import firedrake as fe
import fempy.models.enthalpy_porosity


class Model(fempy.models.enthalpy_porosity.Model):
    
    def __init__(self, *args, meshsize, **kwargs):
        
        self.meshsize = meshsize
        
        self.hot_wall_temperature = fe.Constant(1.)
    
        self.cold_wall_temperature = fe.Constant(-0.01)
        
        self.topwall_heatflux = fe.Constant(0.)
        
        super().__init__(*args, **kwargs)
        
        Ra = 3.27e5
        
        Pr = 56.2
        
        self.grashof_number.assign(Ra/Pr)
        
        self.prandtl_number.assign(Pr)
        
        self.stefan_number.assign(0.045)
        
        self.liquidus_temperature.assign(0.)
        
        self.topwall_heatflux_postswitch = 0.
        
        self.topwall_heatflux_switchtime = 40. + 2.*self.time_tolerance
        
    def init_mesh(self):
    
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def init_dirichlet_boundary_conditions(self):
    
        W = self.function_space
        
        self.dirichlet_boundary_conditions = [
            fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
            fe.DirichletBC(W.sub(2), self.hot_wall_temperature, 1),
            fe.DirichletBC(W.sub(2), self.cold_wall_temperature, 2)]
            
    def init_problem(self):
        
        q = self.topwall_heatflux
        
        _, _, psi_T = fe.TestFunctions(self.function_space)
        
        ds = fe.ds(domain = self.mesh, subdomain_id = 4)
        
        r = self.weak_form_residual*self.integration_measure + psi_T*q*ds
        
        u = self.solution
        
        self.problem = fe.NonlinearVariationalProblem(
            r, u, self.dirichlet_boundary_conditions, fe.derivative(r, u))
            
    def initial_values(self):
        
        return fe.interpolate(
            fe.Expression(
                (0., 0., 0., self.cold_wall_temperature.__float__()),
                element = self.element),
            self.function_space)
            
    def solve(self):
    
        if self.time.__float__() >  \
                (self.topwall_heatflux_switchtime - self.time_tolerance):
            
            self.topwall_heatflux.assign(
                self.topwall_heatflux_postswitch)
    
        super().solve()
        