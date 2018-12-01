import firedrake as fe 
import fempy.mms
import fempy.models.convection_coupled_phasechange


class Model(fempy.models.convection_coupled_phasechange.Model):

    def __init__(self, meshsize):
    
        self.meshsize = meshsize
        
        super().__init__()
        
        self.rayleigh_number.assign(1.e6)
        
        self.prandtl_number.assign(50.)
        
        self.stefan_number.assign(0.05)
        
        self.phase_interface_smoothing.assign(1./32.)
        
    def init_mesh(self):
        
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def init_integration_measure(self):
        
        self.integration_measure = fe.dx(degree = 4)
        
    def strong_form_residual(self, solution):
        
        gamma = self.pressure_penalty_factor
        
        mu_S = self.solid_dynamic_viscosity
        
        mu_L = fe.Constant(1.)
        
        Ra = self.rayleigh_number
        
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        t = self.time
        
        grad, dot, div, sym, diff = fe.grad, fe.dot, fe.div, fe.sym, fe.diff
        
        p, u, T = solution
        
        b = self.buoyancy(T)
        
        phi = self.semi_phasefield(T)
        
        mu = mu_L + phi*(mu_S - mu_L)
        
        r_p = div(u) + gamma*p
        
        r_u = grad(u)*u + grad(p) - 2.*div(mu*sym(grad(u))) + b
        
        r_T = diff(T, t) - 1./Ste*diff(phi, t) + div(T*u) - 1./Pr*div(grad(T))
        
        return r_p, r_u, r_T
        
    def init_manufactured_solution(self):
        
        pi, sin, cos = fe.pi, fe.sin, fe.cos
        
        x = fe.SpatialCoordinate(self.mesh)
        
        t = self.time
        
        t_f = fe.Constant(1.)
        
        T = cos(pi*(2.*t/t_f - 1.))*(1. - sin(x[0])*sin(2.*x[1]))
        
        phi = self.semi_phasefield(T)
        
        u0 = (1. - phi)*pow(t/t_f, 2)*sin(2.*x[0])*sin(x[1])
        
        u1 = (1. - phi)*pow(t/t_f, 2)*sin(x[0])*sin(2.*x[1])
        
        ihat, jhat = self.unit_vectors()
        
        u = u0*ihat + u1*jhat
        
        p = -0.5*(u0**2 + u1**2)
        
        self.manufactured_solution = p, u, T
        
    def init_initial_values(self):
        """ For other models in fempy we have been able to automatically 
        re-use the expressions from `self.init_manufactured_solution`, 
        but `firedrake.interpolate` does not work for this 
        in the case of mixed finite element functions. 
        So unfortunately we have to manually write the string expressions.
        """
        T = "cos(pi*(2.*t/t_f - 1.))*(1. - sin(x[0])*sin(2.*x[1]))"
        
        phi = "0.5*(1. + tanh((T_L - " + T + ")/s))"
            
        u0 = "(1. - " + phi + ")*pow(t/t_f, 2)*sin(2.*x[0])*sin(x[1])"
            
        u1 = "(1. - " + phi + ")*pow(t/t_f, 2)*sin(x[0])*sin(2.*x[1])"
            
        p = "-0.5*(pow(" + u0 + ", 2) + pow(" + u1 + ", 2))"
        
        self.initial_values = [fe.interpolate(
            fe.Expression(
                (p, u0, u1, T),
                t = self.time,
                t_f = 1.,
                T_L = self.liquidus_temperature,
                s = self.phase_interface_smoothing),
            self.function_space),]
        
        
def test__verify_spatial_convergence_order_via_mms(
        grid_sizes = (4, 8, 16, 32),
        timestep_size = 1./64.,
        tolerance = 0.1):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = Model,
        expected_order = 2,
        grid_sizes = grid_sizes,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 1.)
        