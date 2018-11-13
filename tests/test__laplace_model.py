import firedrake as fe 
import fem
import sys


def test__verify_convergence_order_via_mms():

    def strong_form_residual(solution):
    
        div, grad, = fe.div, fe.grad
        
        u = solution
        
        return div(grad(u))
    
    def manufactured_solution(mesh):
        
        sin, pi = fe.sin, fe.pi
        
        x = fe.SpatialCoordinate(mesh)
        
        return sin(2.*pi*x[0])*sin(pi*x[1])
    
    fem.mms.verify_convergence_order(
        Model = fem.laplace_model.LaplaceModel,
        expected_order = 2,
        strong_form_residual = strong_form_residual,
        manufactured_solution = manufactured_solution,
        grid_sizes = (8, 16, 32), 
        tolerance = 0.1)
    
    
if __name__ == "__main__":

    print("Using Python " + sys.version)

    print("Using " + fe.__name__ + "-" + fe.__version__)
    
    test__verify_convergence_order_via_mms()
    