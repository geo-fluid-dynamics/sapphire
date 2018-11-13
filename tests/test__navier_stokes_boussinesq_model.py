import firedrake as fe 
import fem
import sys


def test__verify_convergence_order_via_MMS():

    mu = 0.1
    
    Ra = 10.
    
    Pr = 0.7
    
    ihat, jhat = fe.unit_vector(0, 2), fe.unit_vector(1, 2)
    
    ghat = -jhat
    
    gamma = 1.e-7
    
    residual_parameters = {
        "dynamic_viscosity": mu, 
        "rayleigh_number": Ra,
        "prandtl_number": Pr,
        "gravity_direction": ghat,
        "pressure_penalty_factor": gamma}
    
    def strong_form_residual(solution, mesh):
        
        grad, dot, div, sym = fe.grad, fe.dot, fe.div, fe.sym
        
        p, u, T = solution
        
        r_p = div(u) + gamma*p
        
        r_u = grad(u)*u + grad(p) - 2.*div(mu*sym(grad(u))) + Ra/Pr*T*ghat
        
        r_T = dot(u, grad(T)) - 1./Pr*div(grad(T))
        
        return r_p, r_u, r_T
    
    def manufactured_solution(mesh):
        
        sin, pi = fe.sin, fe.pi
        
        x = fe.SpatialCoordinate(mesh)
        
        u0 = sin(2.*pi*x[0])*sin(pi*x[1])
        
        u1 = sin(pi*x[0])*sin(2.*pi*x[1])
        
        u = u0*ihat + u1*jhat
        
        p = -0.5*(u0**2 + u1**2)
        
        T = sin(2.*pi*x[0])*sin(pi*x[1])
        
        return p, u, T
    
    fem.mms.verify_convergence_order(
        Model = \
            fem.models.navier_stokes_boussinesq_model.NavierStokesBoussinesqModel,
        expected_order = 2,
        strong_form_residual = strong_form_residual,
        manufactured_solution = manufactured_solution,
        residual_parameters = residual_parameters,
        grid_sizes = (2, 4, 8, 16, 32),
        quadrature_degree = 4,
        tolerance = 0.1)
    
    
if __name__ == "__main__":

    print("Using Python " + sys.version)

    print("Using " + fe.__name__ + "-" + fe.__version__)
    
    test__verify_convergence_order_via_MMS()
    