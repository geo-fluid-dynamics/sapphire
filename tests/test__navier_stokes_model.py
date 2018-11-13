import firedrake as fe 
import fem
import sys


def test__verify_convergence_order_via_MMS():

    def strong_form_residual(solution):
    
        grad, dot, div, sym = fe.grad, fe.dot, fe.div, fe.sym
        
        u, p = solution
        
        r_u = grad(u)*u + grad(p) - 2.*div(sym(grad(u)))
        
        r_p = div(u)
        
        return r_u, r_p
    
    def manufactured_solution(mesh):
        
        sin, pi = fe.sin, fe.pi
        
        x = fe.SpatialCoordinate(mesh)
        
        ihat, jhat = fe.unit_vector(0, 2), fe.unit_vector(1, 2)
        
        u = sin(2.*pi*x[0])*sin(pi*x[1])*ihat + sin(pi*x[0])*sin(2.*pi*x[1])*jhat
        
        p = -0.5*(u[0]**2 + u[1]**2)
        
        return u, p
    
    fem.mms.verify_convergence_order(
        Model = fem.models.navier_stokes_model.NavierStokesModel,
        expected_order = 2,
        strong_form_residual = strong_form_residual,
        manufactured_solution = manufactured_solution,
        grid_sizes = (8, 16, 32), 
        tolerance = 0.1)
    
    
if __name__ == "__main__":

    print("Using Python " + sys.version)

    print("Using " + fe.__name__ + "-" + fe.__version__)
    
    test__verify_convergence_order_via_MMS()
    