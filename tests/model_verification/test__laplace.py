import firedrake as fe 
import fempy.mms
import fempy.models.laplace as model_module

    
def manufactured_solution(model):
    
    sin, pi = fe.sin, fe.pi
    
    x = fe.SpatialCoordinate(model.mesh)[0]
    
    return sin(2.*pi*x)
    
    
def test__verify_convergence_order_via_mms(
        mesh_sizes = (8, 16, 32), tolerance = 0.1):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        model_module = model_module,
        model_constructor_kwargs = {"quadrature_degree": 2, "element_degree": 1},
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitIntervalMesh(n) for n in mesh_sizes],
        expected_order = 2,
        tolerance = tolerance)
    