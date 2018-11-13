import firedrake as fe 
import fem
import sys


def test__verify_convergence_order_via_mms():

    sin, pi = fe.sin, fe.pi
    
    ihat, jhat = fe.unit_vector(0, 2), fe.unit_vector(1, 2)
    
    def a(x):
    
        return sin(2.*pi*x[0])*sin(4.*pi*x[1])*ihat + sin(pi*x[0])*sin(2.*pi*x[1])*jhat
    
    nu = 0.1
    
    residual_parameters = {"velocity": a, "viscosity": nu}
    
    def strong_form_residual(solution, mesh):
        
        x = fe.SpatialCoordinate(mesh)
        
        a = residual_parameters["velocity"](x)
        
        nu = residual_parameters["viscosity"]
        
        dot, grad, div = fe.dot, fe.grad, fe.div
        
        u = solution
        
        return dot(a, grad(u)) - div(nu*grad(u))
    
    def manufactured_solution(mesh):
        
        sin, pi = fe.sin, fe.pi
        
        x = fe.SpatialCoordinate(mesh)
        
        return sin(2.*pi*x[0])*sin(pi*x[1])
    
    fem.mms.verify_convergence_order(
        Model = fem.models.convection_diffusion_model.ConvectionDiffusionModel,
        residual_parameters = residual_parameters,
        expected_order = 2,
        strong_form_residual = strong_form_residual,
        manufactured_solution = manufactured_solution,
        grid_sizes = (8, 16, 32),
        quadrature_degree = 2,
        tolerance = 0.1)
    
    
if __name__ == "__main__":

    print("Using Python " + sys.version)

    print("Using " + fe.__name__ + "-" + fe.__version__)
    
    test__verify_convergence_order_via_mms()
    