import firedrake as fe 
import fem
import sys


def test__verify_convergence_order_via_MMS():
    
    def strong_form_residual(model):
        
        grad, dot, div, sym = fe.grad, fe.dot, fe.div, fe.sym
        
        p, u, T = model.manufactured_solution()
        
        r_p = div(u) + gamma*p
        
        r_u = grad(u)*u + grad(p) - 2.*div(mu*sym(grad(u))) + Ra/Pr*T*ghat
        
        r_T = dot(u, grad(T)) - 1./Pr*div(grad(T))
        
        return r_p, r_u, r_T
    
    def manufactured_solution(model):
        
        sin, pi = fe.sin, fe.pi
        
        x = fe.SpatialCoordinate(model.mesh)
        
        u0 = sin(2.*pi*x[0])*sin(pi*x[1])
        
        u1 = sin(pi*x[0])*sin(2.*pi*x[1])
        
        ihat, jhat = model.unit_vectors()
        
        u = u0*ihat + u1*jhat
        
        p = -0.5*(u0**2 + u1**2)
        
        T = sin(2.*pi*x[0])*sin(pi*x[1])
        
        return p, u, T
        
    def set_parameters(model):
    
        model.dynamic_viscosity.assign(0.1)
    
        model.rayleigh_number.assign(10.)
        
        model.prandtl_number(0.7)
        
        ihat, jhat = model.unit_vectors()
        
        model.gravity_direction.assign(-jhat)
        
        model.pressure_penalty_factor.assign(1.e-7)
    
    fem.mms.verify_order_of_accuracy(
        Model = \
            fem.models.navier_stokes_boussinesq_model.NavierStokesBoussinesqModel,
        expected_spatial_order = 2,
        strong_form_residual = strong_form_residual,
        manufactured_solution = manufactured_solution,
        set_parameters = set_parameters,
        residual_parameters = residual_parameters,
        grid_sizes = (2, 4, 8, 16, 32),
        quadrature_degree = 4,
        tolerance = 0.1)
    
    
if __name__ == "__main__":

    print("Using Python " + sys.version)

    print("Using " + fe.__name__ + "-" + fe.__version__)
    
    test__verify_convergence_order_via_MMS()
    