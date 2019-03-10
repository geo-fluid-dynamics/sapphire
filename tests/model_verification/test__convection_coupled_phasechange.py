import firedrake as fe 
import fempy.mms
from fempy.models import convection_coupled_phasechange as model_module


def manufactured_solution(model):

    pi, sin, cos, exp = fe.pi, fe.sin, fe.cos, fe.exp
    
    x = fe.SpatialCoordinate(model.mesh)
    
    t = model.time
    
    t_f = fe.Constant(1.)
    
    ihat, jhat = fempy.model.unit_vectors(model.mesh)
    
    u = exp(t)*sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
        exp(t)*sin(pi*x[0])*sin(2.*pi*x[1])*jhat
    
    p = -sin(pi*x[0])*sin(2.*pi*x[1])
    
    T = 0.5*sin(2.*pi*x[0])*sin(pi*x[1])*(1. - exp(-t**2))
    
    return p, u, T
    
    
def test__verify__second_order_spatial_convergence__via_mms(
        model_constructor_kwargs = {
            "quadrature_degree": 4,
            "element_degree": 1,
            "time_stencil_size": 3},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "heat_capacity_solid_to_liquid_ratio": 0.500,
            "thermal_conductivity_solid_to_liquid_ratio": 2.14/0.561,
            "smoothing": 1./16.},
        mesh_sizes = (2, 4, 8),
        timestep_size = 1./128.,
        tolerance = 0.05):
    
    rt = model_constructor_kwargs["time_stencil_size"] - 1
    
    fempy.mms.verify_spatial_order_of_accuracy(
        model_module = model_module,
        manufactured_solution = manufactured_solution,
        model_constructor_kwargs = model_constructor_kwargs,
        meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
        parameters = parameters,
        expected_order = 2,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 0.5,
        outdir_path_prefix = "output/mms/space/rt{0}_Deltat{1}/".format(
            rt, timestep_size))
        
        
def test__verify__second_order_temporal_convergence__via_mms(
        model_constructor_kwargs = {
            "quadrature_degree": 4,
            "element_degree": 1,
            "time_stencil_size": 3},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "heat_capacity_solid_to_liquid_ratio": 0.500,
            "thermal_conductivity_solid_to_liquid_ratio": 2.14/0.561,
            "smoothing": 1./16.},
        meshsize = 24,
        timestep_sizes = (1./8., 1./16., 1./32.),
        tolerance = 0.4):
    
    rx = model_constructor_kwargs["element_degree"] + 1
    
    mesh = fe.UnitSquareMesh(meshsize, meshsize)
    
    h = mesh.cell_sizes((0.,)*mesh.geometric_dimension())
    
    fempy.mms.verify_temporal_order_of_accuracy(
        model_module = model_module,
        manufactured_solution = manufactured_solution,
        model_constructor_kwargs = model_constructor_kwargs,
        mesh = mesh,
        parameters = parameters,
        expected_order = 2,
        tolerance = tolerance,
        timestep_sizes = timestep_sizes,
        endtime = 0.5,
        outdir_path_prefix = "output/mms/time/rx{0}_h{1}/".format(
            rx, h))

        
def test__verify__third_order_spatial_convergence__via_mms(
        model_constructor_kwargs = {
            "quadrature_degree": 2,
            "element_degree": 2,
            "time_stencil_size": 4},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "heat_capacity_solid_to_liquid_ratio": 0.500,
            "thermal_conductivity_solid_to_liquid_ratio": 2.14/0.561,
            "smoothing": 1./16.},
        mesh_sizes = (3, 6, 12, 24),
        timestep_size = 1./64.,
        tolerance = 0.32):
    
    rt = model_constructor_kwargs["time_stencil_size"] - 1
    
    fempy.mms.verify_spatial_order_of_accuracy(
        model_module = model_module,
        manufactured_solution = manufactured_solution,
        model_constructor_kwargs = model_constructor_kwargs,
        meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
        parameters = parameters,
        expected_order = 3,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 0.32,
        outdir_path_prefix = "output/mms/space/rt{0}_Deltat{1}/".format(
            rt, timestep_size))
        
        
def test__verify__third_order_temporal_convergence__via_mms(
        model_constructor_kwargs = {
            "quadrature_degree": 2,
            "element_degree": 2,
            "time_stencil_size": 4},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "heat_capacity_solid_to_liquid_ratio": 0.500,
            "thermal_conductivity_solid_to_liquid_ratio": 2.14/0.561,
            "smoothing": 1./16.},
        meshsize = 16,
        timestep_sizes = (1./2., 1./4., 1./8., 1./16.),
        tolerance = 0.36):
    
    rx = model_constructor_kwargs["element_degree"] + 1
    
    mesh = fe.UnitSquareMesh(meshsize, meshsize)
    
    h = mesh.cell_sizes((0.,)*mesh.geometric_dimension())
    
    fempy.mms.verify_temporal_order_of_accuracy(
        model_module = model_module,
        manufactured_solution = manufactured_solution,
        model_constructor_kwargs = model_constructor_kwargs,
        mesh = mesh,
        parameters = parameters,
        expected_order = 3,
        tolerance = tolerance,
        timestep_sizes = timestep_sizes,
        endtime = 0.5,
        outdir_path_prefix = "output/mms/time/rx{0}_h{1}/".format(
            rx, h))
        