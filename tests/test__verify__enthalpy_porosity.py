import firedrake as fe 
import fempy.mms
import fempy.models.enthalpy_porosity


class VerifiableModel(fempy.models.enthalpy_porosity.Model):

    def __init__(self, *args, meshsize, **kwargs):
    
        self.meshsize = meshsize
        
        super().__init__(*args, **kwargs)
        
    def init_mesh(self):
        
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def strong_form_residual(self, solution):
        
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        t = self.time
        
        grad, dot, div, sym, diff = fe.grad, fe.dot, fe.div, fe.sym, fe.diff
        
        p, u, T = solution
        
        b = self.buoyancy(T)
        
        d = self.solid_velocity_relaxation(T)
        
        phil = self.porosity(T)
        
        r_p = div(u)
        
        r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(sym(grad(u))) \
            + b + d*u
        
        r_T = diff(T + 1./Ste*phil, t) + dot(u, grad(T)) - 1./Pr*div(grad(T))
        
        return r_p, r_u, r_T
        
    def init_manufactured_solution(self):
        
        pi, sin, cos, exp = fe.pi, fe.sin, fe.cos, fe.exp
        
        x = fe.SpatialCoordinate(self.mesh)
        
        t = self.time
        
        t_f = fe.Constant(1.)
        
        ihat, jhat = self.unit_vectors()
        
        u = exp(t)*sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
            exp(t)*sin(pi*x[0])*sin(2.*pi*x[1])*jhat
        
        p = -sin(pi*x[0])*sin(2.*pi*x[1])
        
        T = 0.5*sin(2.*pi*x[0])*sin(pi*x[1])*(1. - exp(-t**2))
        
        self.manufactured_solution = p, u, T
    
    
def test__verify__second_order_spatial_convergence__via_mms(
        constructor_kwargs = {
            "quadrature_degree": 4,
            "spatial_order": 2,
            "temporal_order": 4},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "smoothing": 1./16.},
        mesh_sizes = (5, 10, 20),
        timestep_size = 1./128.,
        tolerance = 0.23):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableModel,
        constructor_kwargs = constructor_kwargs,
        parameters = parameters,
        expected_order = 2,
        mesh_sizes = mesh_sizes,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 0.5,
        plot_solution = False,
        plot_errors = False,
        report = False)
        
        
def test__verify__second_order_temporal_convergence__via_mms(
        constructor_kwargs = {
            "quadrature_degree": 4,
            "spatial_order": 2,
            "temporal_order": 2},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "smoothing": 1./16.},
        meshsize = 20,
        timestep_sizes = (1./8., 1./16., 1./32.),
        tolerance = 0.34):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = VerifiableModel,
        constructor_kwargs = constructor_kwargs,
        parameters = parameters,
        expected_order = 2,
        meshsize = meshsize,
        tolerance = tolerance,
        timestep_sizes = timestep_sizes,
        endtime = 0.5,
        plot_solution = False,
        plot_errors = False,
        report = False)

        
def test__verify__third_order_spatial_convergence__via_mms(
        constructor_kwargs = {
            "quadrature_degree": 2,
            "spatial_order": 3,
            "temporal_order": 3},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "smoothing": 1./16.},
        mesh_sizes = (3, 6, 12, 24),
        timestep_size = 1./64.,
        tolerance = 0.32):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableModel,
        constructor_kwargs = constructor_kwargs,
        parameters = parameters,
        expected_order = 3,
        mesh_sizes = mesh_sizes,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 0.32,
        plot_solution = False,
        plot_errors = False,
        report = False)
        
        
def test__verify__third_order_temporal_convergence__via_mms(
        constructor_kwargs = {
            "quadrature_degree": 8,
            "spatial_order": 3,
            "temporal_order": 3},
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "smoothing": 1./16.},
        meshsize = 16,
        timestep_sizes = (1./2., 1./4., 1./8., 1./16.),
        tolerance = 0.02):
    
    fempy.mms.verify_temporal_order_of_accuracy(
        Model = VerifiableModel,
        constructor_kwargs = constructor_kwargs,
        parameters = parameters,
        expected_order = 3,
        meshsize = meshsize,
        tolerance = tolerance,
        timestep_sizes = timestep_sizes,
        endtime = 0.5,
        plot_solution = False,
        plot_errors = False,
        report = False)
        