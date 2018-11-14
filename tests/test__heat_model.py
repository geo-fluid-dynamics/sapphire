import firedrake as fe 
import fem
import sys


t = fe.variable(0.)

def test__verify_convergence_order_via_mms():

    alpha = fe.Constant(3.)
    
    def strong_form_residual(solution, mesh):
        
        u = solution
        
        diff, div, grad = fe.diff, fe.div, fe.grad
        
        return diff(u, t) - alpha*div(grad(u))
    
    def manufactured_solution(mesh):
        
        sin, pi, exp = fe.sin, fe.pi, fe.exp
        
        x = fe.SpatialCoordinate(mesh)
        
        return sin(2.*pi*x[0])*sin(pi*x[1])*exp(-t**2)
    
    fem.mms.verify_orders_of_accuracy(
        Model = fem.models.heat_model.HeatModel,
        residual_parameters = {"thermal_diffusivity": alpha},
        strong_form_residual = strong_form_residual,
        manufactured_solution = manufactured_solution,
        expected_spatial_order = 2,
        grid_sizes = (8, 16, 32),
        expected_temporal_order = 1,
        endtime = 1.,
        timestep_sizes = (1., 1./2., 1./4.),
        quadrature_degree = 2,
        tolerance = 0.1)
