import firedrake as fe 
import fempy.models.navier_stokes


def manufactured_solution(model):
    
    sin, pi = fe.sin, fe.pi
    
    x = fe.SpatialCoordinate(model.mesh)
    
    ihat, jhat = model.unit_vectors()
    
    u = sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
        sin(pi*x[0])*sin(2.*pi*x[1])*jhat
    
    p = -0.5*(u[0]**2 + u[1]**2)
    
    return u, p
        
        
def test__verify_convergence_order_via_mms(
        mesh_sizes = (16, 32), tolerance = 0.1):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = fempy.models.navier_stokes.Model,
        meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
        model_constructor_kwargs = {"quadrature_degree": 4, "element_degree": 1},
        manufactured_solution = manufactured_solution,
        expected_order = 2,
        tolerance = tolerance)
    