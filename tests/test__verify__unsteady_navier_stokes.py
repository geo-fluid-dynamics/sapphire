import firedrake as fe 
import fempy.models.unsteady_navier_stokes


def manufactured_solution(model):
    
    exp, sin, pi = fe.exp, fe.sin, fe.pi
    
    x = fe.SpatialCoordinate(model.mesh)
    
    t = model.time
    
    ihat, jhat = fempy.model.unit_vectors(model.mesh)
    
    u = exp(t)*(sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
        sin(pi*x[0])*sin(2.*pi*x[1])*jhat)
    
    p = -0.5*(u[0]**2 + u[1]**2)
    
    return u, p
    
    
def test__verify_spatial_convergence__second_order__via_mms(
        mesh_sizes = (3, 6, 12, 24),
        timestep_size = 1./32.,
        tolerance = 0.3):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = fempy.models.unsteady_navier_stokes.Model,
        weak_form_residual = \
            fempy.models.unsteady_navier_stokes._variational_form_residual,
        strong_form_residual = \
            fempy.models.unsteady_navier_stokes._strong_form_residual,
        meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
        model_constructor_kwargs = {
            "quadrature_degree": 4,
            "element_degree": 1,
            "time_stencil_size": 2},
        manufactured_solution = manufactured_solution,
        expected_order = 2,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 1.)
    
 
def test__verify_temporal_convergence__first_order__via_mms(
        meshsize = 32,
        timestep_sizes = (1./2., 1./4., 1./8.),
        tolerance = 0.1):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = fempy.models.unsteady_navier_stokes.Model,
        weak_form_residual = \
            fempy.models.unsteady_navier_stokes._variational_form_residual,
        strong_form_residual = \
            fempy.models.unsteady_navier_stokes._strong_form_residual,
        mesh = fe.UnitSquareMesh(meshsize, meshsize),
        model_constructor_kwargs = {
            "quadrature_degree": 4,
            "element_degree": 1,
            "time_stencil_size": 2},
        manufactured_solution = manufactured_solution,
        expected_order = 1,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance)
    