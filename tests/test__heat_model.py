import firedrake as fe 
import fem
import sys


def test__verify_convergence_order_via_mms():
    
    class Model(fem.models.heat_model.HeatModel):
    
        def __init__(self, gridsize = 4):
        
            self.gridsize = gridsize
            
            super().__init__()
            
            self.thermal_diffusivity.assign(3.)
            
        def mesh(self):
        
            return fe.UnitSquareMesh(self.gridsize, self.gridsize)
        
        def strong_form_residual(self):
            
            u = self.manufactured_solution()
            
            t = self.ufl_time
            
            diff, div, grad = fe.diff, fe.div, fe.grad
            
            return diff(u, t) - alpha*div(grad(u))
    
        def manufactured_solution(self):
            
            x = fe.SpatialCoordinate(self.mesh)
            
            t = self.ufl_time
            
            sin, pi, exp = fe.sin, fe.pi, fe.exp
            
            return sin(2.*pi*x[0])*sin(pi*x[1])*exp(-t)
        
    fem.mms.verify_order_of_accuracy(
        Model = Model,
        expected_spatial_order = 2,
        grid_sizes = (8, 16, 32),
        expected_temporal_order = 1,
        endtime = 1.,
        timestep_sizes = (1., 1./2., 1./4.),
        quadrature_degree = 2,
        tolerance = 0.1)
