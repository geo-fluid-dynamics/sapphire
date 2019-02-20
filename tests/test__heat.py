import firedrake as fe 
import fempy.models.heat


class VerifiableModel(fempy.models.heat.Model):
    
    def __init__(self,
            quadrature_degree,
            spatial_order,
            temporal_order,
            meshsize):
    
        self.meshsize = meshsize
        
        super().__init__(
            quadrature_degree = quadrature_degree,
            spatial_order = spatial_order,
            temporal_order = temporal_order)
        
    def init_mesh(self):
    
        self.mesh = fe.UnitIntervalMesh(self.meshsize)
        
    def strong_form_residual(self, solution):
        
        u = solution
        
        t = self.time
        
        diff, div, grad = fe.diff, fe.div, fe.grad
        
        return diff(u, t) - div(grad(u))
    
    def init_manufactured_solution(self):
        
        x = fe.SpatialCoordinate(self.mesh)[0]
        
        t = self.time
        
        sin, pi, exp = fe.sin, fe.pi, fe.exp
        
        self.manufactured_solution = sin(2.*pi*x)*exp(-pow(t, 2))
        
    def init_solver(self, solver_parameters = {"ksp_type": "cg"}):
        
        self.solver = fe.NonlinearVariationalSolver(
            self.problem, solver_parameters = solver_parameters)

        
def test__verify_spatial_convergence__second_order__via_mms(
        mesh_sizes = (4, 8, 16, 32),
        timestep_size = 1./64.,
        tolerance = 0.1):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableModel,
        constructor_kwargs = {
            "quadrature_degree": None, "spatial_order": 2, "temporal_order": 1},
        expected_order = 2,
        mesh_sizes = mesh_sizes,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 1.,
        plot_solution = False,
        report = False)
        
        
def test__verify_temporal_convergence__first_order__via_mms(
        meshsize = 256,
        timestep_sizes = (1./4., 1./8., 1./16., 1./32.),
        tolerance = 0.1):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = VerifiableModel,
        constructor_kwargs = {
            "quadrature_degree": None, "spatial_order": 2, "temporal_order": 1},
        expected_order = 1,
        meshsize = meshsize,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance,
        plot_solution = False,
        report = False)
    
    
def test__verify_temporal_convergence__second_order__via_mms(
        meshsize = 128,
        timestep_sizes = (
            1./4., 1./8., 1./16., 1./32., 1./64., 1./128.),
        tolerance = 0.1):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = VerifiableModel,
        constructor_kwargs = {
            "quadrature_degree": None, "spatial_order": 3, "temporal_order": 2},
        expected_order = 2,
        meshsize = meshsize,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance,
        plot_solution = False,
        report = False)
        
        
def test__verify_temporal_convergence__third_order__via_mms(
        meshsize = 128,
        timestep_sizes = (1./4., 1./8., 1./16., 1./32.),
        tolerance = 0.1):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = VerifiableModel,
        constructor_kwargs = {
            "quadrature_degree": None, "spatial_order": 3, "temporal_order": 3},
        expected_order = 3,
        meshsize = meshsize,
        endtime = 1.,
        timestep_sizes = timestep_sizes,
        tolerance = tolerance,
        plot_solution = False,
        report = False)
        