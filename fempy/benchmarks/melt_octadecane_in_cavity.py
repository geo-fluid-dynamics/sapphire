import firedrake as fe
import fempy.models.enthalpy_porosity


def initial_values(model):
    
    return fe.interpolate(
        fe.Expression(
            (0., 0., 0., model.cold_wall_temperature.__float__()),
            element = model.element),
        model.function_space)
        
        
def dirichlet_boundary_conditions(model):

    W = model.function_space
    
    return [
        fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
        fe.DirichletBC(W.sub(2), model.hot_wall_temperature, 1),
        fe.DirichletBC(W.sub(2), model.cold_wall_temperature, 2)]
        
        
class Model(fempy.models.enthalpy_porosity.Model):
    
    def __init__(self, *args, meshsize, **kwargs):
        
        self.hot_wall_temperature = fe.Constant(1.)
    
        self.cold_wall_temperature = fe.Constant(-0.01)
        
        self.topwall_heatflux = fe.Constant(0.)
        
        super().__init__(
            *args,
            mesh = fe.UnitSquareMesh(meshsize, meshsize),
            initial_values = initial_values,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            **kwargs)
        
        q = self.topwall_heatflux

        _, _, psi_T = fe.TestFunctions(self.function_space)

        ds = fe.ds(domain = self.mesh, subdomain_id = 4)

        self.variational_form_residual += psi_T*q*ds
        
        self.reset_problem_and_solver()
        
        self.topwall_heatflux_postswitch = 0.
        
        self.topwall_heatflux_switchtime = 40. + 2.*self.time_tolerance
        
        Ra = 3.27e5
        
        Pr = 56.2
        
        self.grashof_number = self.grashof_number.assign(Ra/Pr)
        
        self.prandtl_number = self.prandtl_number.assign(Pr)
        
        self.stefan_number = self.stefan_number.assign(0.045)
        
        self.liquidus_temperature = self.liquidus_temperature.assign(0.)
        
    def solve(self):
    
        if self.time.__float__() >  \
                (self.topwall_heatflux_switchtime - self.time_tolerance):
            
            self.topwall_heatflux.assign(
                self.topwall_heatflux_postswitch)
    
        self.solution, self.snes_iteration_count = super().solve()
        
        return self.solution, self.snes_iteration_count
        