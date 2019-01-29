import firedrake as fe 
import fempy.mms
import fempy.models.enthalpy_porosity
import fempy.benchmarks.melting_octadecane


def test__melting_octadecane_benchmark__regression():
    
    endtime, expected_liquid_area, tolerance = 30., 0.24, 0.01
    
    nx = 32
    
    Delta_t = 10.
    
    model = fempy.benchmarks.melting_octadecane.Model(meshsize = nx)
    
    model.timestep_size.assign(Delta_t)
    
    s = 1./256.
    
    model.smoothing.assign(s)
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "melting_octadecane/" + "nx" + str(nx) + "_Deltat" + str(Delta_t) 
        + "_s" + str(s) + "_tf" + str(endtime) + "/")
        
    model.run(endtime = endtime, plot = False)
    
    p, u, T = model.solution.split()
    
    phil = model.porosity(T)
    
    liquid_area = fe.assemble(phil*fe.dx)
    
    print("Liquid area = " + str(liquid_area))
    
    assert(abs(liquid_area - expected_liquid_area) < tolerance)
    
    phil_h = fe.interpolate(phil, model.function_space.sub(2))
    
    max_phil = phil_h.vector().max()
    
    print("Maximum phil = " + str(max_phil))
    
    assert(abs(max_phil - 1.) < tolerance)
    
    
def test__melting_octadecane_benchmark_with_darcy_resistance__regression():
    
    endtime, expected_liquid_area, tolerance = 30., 0.24, 0.01
    
    D = 1.e12
    
    model = fempy.benchmarks.melting_octadecane.ModelWithDarcyResistance(meshsize = 32)
    
    model.timestep_size.assign(10.)
    
    model.smoothing.assign(1./256.)
    
    model.darcy_resistance_factor.assign(D)
    
    model.output_directory_path = model.output_directory_path.joinpath(
        "melting_octadecane_with_darcy_resistance/D" + str(D) + "/")
        
    model.run(endtime = endtime, plot = False)
    
    p, u, T = model.solution.split()
    
    phil = model.porosity(T)
    
    liquid_area = fe.assemble(phil*fe.dx)
    
    print("Liquid area = " + str(liquid_area))
    
    assert(abs(liquid_area - expected_liquid_area) < tolerance)
    
    phil_h = fe.interpolate(phil, model.function_space.sub(2))
    
    max_phil = phil_h.vector().max()
    
    print("Maximum phil = " + str(max_phil))
    
    assert(abs(max_phil - 1.) < tolerance)


class VerifiableModel(fempy.models.enthalpy_porosity.Model):

    def __init__(self, meshsize):
    
        self.meshsize = meshsize
        
        super().__init__()
        
    def init_mesh(self):
        
        self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
        
    def init_integration_measure(self):
        
        self.integration_measure = fe.dx(degree = 4)
        
    def strong_form_residual(self, solution):
        
        gamma = self.pressure_penalty_factor
        
        mu_s = self.solid_dynamic_viscosity
        
        mu_l = self.liquid_dynamic_viscosity
        
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        t = self.time
        
        grad, dot, div, sym, diff = fe.grad, fe.dot, fe.div, fe.sym, fe.diff
        
        p, u, T = solution
        
        b = self.buoyancy(T)
        
        phil = self.porosity(T)
        
        mu = mu_s + (mu_l - mu_s)*phil
        
        r_p = div(u) + gamma*p
        
        r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(mu*sym(grad(u))) + b
        
        r_T = diff(T, t) + 1./Ste*diff(phil, t) + div(T*u) - 1./Pr*div(grad(T))
        
        return r_p, r_u, r_T
        
    def init_manufactured_solution(self):
        
        pi, sin, cos, exp = fe.pi, fe.sin, fe.cos, fe.exp
        
        x = fe.SpatialCoordinate(self.mesh)
        
        t = self.time
        
        t_f = fe.Constant(1.)
        
        ihat, jhat = self.unit_vectors()
        
        u = exp(t)*sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
            exp(t)*sin(pi*x[0])*sin(2.*pi*x[1])*jhat
        
        p = -0.5*(u[0]**2 + u[1]**2)
        
        T = 0.5*sin(2.*pi*x[0])*sin(pi*x[1])*(1. - 2*exp(-3.*t**2))
        
        self.manufactured_solution = p, u, T
        
    def update_initial_values(self):
        
        for u_m, V in zip(
                self.manufactured_solution, self.function_space):
        
            self.initial_values.assign(fe.interpolate(u_m, V))
        
    def solve(self):
        
        self.solver.parameters["snes_monitor"] = False
        
        super().solve()
            
        print("Solved at time t = " + str(self.time.__float__()))
        
        
def fails__test__verify_spatial_convergence_order_via_mms(
        parameters = {
            "grashof_number": 2.,
            "prandtl_number": 5.,
            "stefan_number": 0.2,
            "smoothing": 1./16.},
        mesh_sizes = (4, 8, 16),
        timestep_size = 1./64.,
        tolerance = 0.2):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableModel,
        parameters = parameters,
        expected_order = 2,
        mesh_sizes = mesh_sizes,
        tolerance = tolerance,
        timestep_size = timestep_size,
        endtime = 1.)
    