import firedrake as fe 
import fem


def test__verify_convergence_order_via_MMS(
        grid_sizes = (16, 32), tolerance = 0.1, quadrature_degree = 4):
    
    class Model(fem.models.navier_stokes_boussinesq.Model):
    
        def __init__(self, gridsize = 4):
        
            self.gridsize = gridsize
            
            super().__init__()
            
            self.dynamic_viscosity.assign(0.1)
        
            self.rayleigh_number.assign(10.)
            
            self.prandtl_number(0.7)
            
            self.pressure_penalty_factor.assign(1.e-7)
            
        def init_mesh(self):
        
            self.mesh = fe.UnitSquareMesh(self.gridsize, self.gridsize)
            
        def init_integration_measure(self):

            self.integration_measure = fe.dx(degree = 4)
            
        def init_manufactured_solution(self):
            
            sin, pi = fe.sin, fe.pi
            
            x = fe.SpatialCoordinate(self.mesh)
            
            u0 = sin(2.*pi*x[0])*sin(pi*x[1])
            
            u1 = sin(pi*x[0])*sin(2.*pi*x[1])
            
            ihat, jhat = self.unit_vectors()
            
            u = u0*ihat + u1*jhat
            
            p = -0.5*(u0**2 + u1**2)
            
            T = sin(2.*pi*x[0])*sin(pi*x[1])
            
            self.manufactured_solution = p, u, T
            
        def strong_form_residual(self, solution):
            
            gamma = self.pressure_penalty_factor
            
            mu = self.dynamic_viscosity
            
            Ra = self.rayleigh_number
            
            Pr = self.prandtl_number
            
            ghat = self.gravity_direction
            
            grad, dot, div, sym = fe.grad, fe.dot, fe.div, fe.sym
            
            p, u, T = solution
            
            r_p = div(u) + gamma*p
            
            r_u = grad(u)*u + grad(p) - 2.*div(mu*sym(grad(u))) + Ra/Pr*T*ghat
            
            r_T = dot(u, grad(T)) - 1./Pr*div(grad(T))
            
            return r_p, r_u, r_T
            
    fem.mms.verify_spatial_order_of_accuracy(
        Model = Model,
        expected_order = 2,
        grid_sizes = grid_sizes,
        tolerance = tolerance)
    