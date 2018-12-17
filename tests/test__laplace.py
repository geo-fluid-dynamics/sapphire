import firedrake as fe 
import fempy.models.laplace


class VerifiableModel(fempy.models.laplace.Model):
        
    def strong_form_residual(self, solution):
    
        div, grad, = fe.div, fe.grad
        
        u = solution
        
        return div(grad(u))

        
class OneDimMMSModel(VerifiableModel):
    
    def __init__(self, meshsize):
        
        self.meshsize = meshsize
        
        super().__init__()
        
        self.integration_measure = fe.dx(degree = 2)
        
    def init_mesh(self):
        
        self.mesh = fe.UnitIntervalMesh(self.meshsize)
        
    def init_manufactured_solution(self):
        
        sin, pi = fe.sin, fe.pi
        
        x = fe.SpatialCoordinate(self.mesh)[0]
        
        self.manufactured_solution = sin(2.*pi*x)

        
def test__verify_convergence_order_via_mms(
        mesh_sizes = (16, 32), tolerance = 0.1):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = OneDimMMSModel,
        expected_order = 2,
        mesh_sizes = mesh_sizes,
        tolerance = tolerance)
    