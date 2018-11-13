import firedrake as fe 
import fem
import sys


t = fe.variable("t")

def test__verify_convergence_order_via_mms():

    def strong_form_residual(solution, mesh):
        
        alpha = self.residual_parameters["thermal_diffusivity"]
        
        u = solution
        
        diff, div, grad = fe.diff, fe.div, fe.grad
        
        return diff(u, t) - alpha*div(grad(u))
    
    def manufactured_solution(mesh):
        
        sin, pi = fe.sin, fe.pi
        
        x = fe.SpatialCoordinate(mesh)
        
        return sin(2.*pi*x[0])*sin(pi*x[1])*exp(-t**2)
    
    fem.mms.verify_convergence_orders(
        Model = fem.models.heat_model.HeatModel,
        expected_space_order = 2,
        expected_time_order = 1,
        strong_form_residual = strong_form_residual,
        manufactured_solution = manufactured_solution,
        grid_sizes = (8, 16, 32),
        timestep_sizes = (1., 1./2., 1./4.),
        quadrature_degree = 2,
        tolerance = 0.1)
    
    
if __name__ == "__main__":

    print("Using Python " + sys.version)

    print("Using " + fe.__name__ + "-" + fe.__version__)
    
    test__verify_convergence_order_via_mms()
