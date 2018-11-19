import firedrake as fe 
import fem


def test__verify_convergence_order_via_mms(
        grid_sizes = (16, 32), tolerance = 0.1, quadrature_degree = 2):
    
    class Model(fem.models.convection_diffusion.Model):
        
        def __init__(self, gridsize = 4):
            
            self.gridsize = gridsize
            
            super().__init__()
                
            self.kinematic_viscosity.assign(0.1)
            
        def init_mesh(self):
            
            self.mesh = fe.UnitSquareMesh(self.gridsize, self.gridsize)
        
        def init_integration_measure(self):

            self.integration_measure = fe.dx(degree = 2)
        
        def init_manufactured_solution(self):
            
            x = fe.SpatialCoordinate(self.mesh)
            
            sin, pi = fe.sin, fe.pi
            
            self.manufactured_solution = sin(2.*pi*x[0])*sin(pi*x[1])
            
            ihat, jhat = self.unit_vectors()
            
            self.advection_velocity = sin(2.*pi*x[0])*sin(4.*pi*x[1])*ihat \
                + sin(pi*x[0])*sin(2.*pi*x[1])*jhat
        
        def strong_form_residual(self, solution):
            
            x = fe.SpatialCoordinate(self.mesh)
            
            u = solution
            
            a = self.advection_velocity
            
            nu = self.kinematic_viscosity
            
            dot, grad, div = fe.dot, fe.grad, fe.div
            
            return dot(a, grad(u)) - div(nu*grad(u))
        
    fem.mms.verify_spatial_order_of_accuracy(
        Model = Model,
        expected_order = 2,
        grid_sizes = grid_sizes,
        tolerance = tolerance,
        quadrature_degree = quadrature_degree)