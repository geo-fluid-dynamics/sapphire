import firedrake as fe 
import fempy.mms
import fempy.models.navier_stokes_boussinesq
import fempy.benchmarks.heat_driven_cavity


class VerifiableModel(fempy.models.navier_stokes_boussinesq.Model):
    
        def __init__(self, meshsize):
        
            self.meshsize = meshsize
            
            super().__init__()
            
        def init_mesh(self):
        
            self.mesh = fe.UnitSquareMesh(self.meshsize, self.meshsize)
            
        def init_integration_measure(self):

            self.integration_measure = fe.dx(degree = 4)
            
        def init_manufactured_solution(self):
            
            sin, pi = fe.sin, fe.pi
            
            x = fe.SpatialCoordinate(self.mesh)
            
            u0 = sin(2.*pi*x[0])*sin(pi*x[1])
            
            u1 = sin(pi*x[0])*sin(2.*pi*x[1])
            
            ihat, jhat = self.unit_vectors()
            
            u = u0*ihat + u1*jhat
            
            p = -0.5*(u0**2 + u1**2)
            
            T = sin(2.*pi*x[0])*sin(pi*x[1])
            
            self.manufactured_solution = p, u, T
            
        def strong_form_residual(self, solution):
            
            mu = self.dynamic_viscosity
            
            Ra = self.rayleigh_number
            
            Pr = self.prandtl_number
            
            ghat = self.gravity_direction
            
            grad, dot, div, sym = fe.grad, fe.dot, fe.div, fe.sym
            
            p, u, T = solution
            
            r_p = div(u)
            
            r_u = grad(u)*u + grad(p) - 2.*div(mu*sym(grad(u))) + Ra/Pr*T*ghat
            
            r_T = dot(u, grad(T)) - 1./Pr*div(grad(T))
            
            return r_p, r_u, r_T
            

def test__verify_convergence_order_via_mms(
        mesh_sizes = (4, 8, 16, 32, 64), 
        tolerance = 0.1, 
        plot_errors = False,
        plot_solution = False):
    
    fempy.mms.verify_spatial_order_of_accuracy(
        Model = VerifiableModel,
        parameters = {
            "dynamic_viscosity": 0.1, 
            "rayleigh_number": 10., 
            "prandtl_number": 0.7},
        expected_order = 2,
        mesh_sizes = mesh_sizes,
        tolerance = tolerance,
        plot_errors = plot_errors,
        plot_solution = plot_solution)
    
    
def verify_scalar_solution_component(
            model, 
            component, 
            coordinates, 
            verified_values, 
            relative_tolerance, 
            absolute_tolerance,
            subcomponent = None):
        """ Verify the scalar values of a specified solution component.
        
        Parameters
        ----------
        model : fempy.Model
        
        component : integer
        
            The solution is often vector-valued and based on a mixed formulation.
            By having the user specify a component to verify with this function,
            we can write the rest of the function quite generally.
            
        coordinates : list of tuples, where each tuple contains a float for each spatial dimension.
        
            Each tuple will be converted to a `Point`.
            
        verified_values : tuple of floats
        
           Point-wise verified values from a benchmark publication.
           
        relative_tolerance : float   
           
           This will be used for asserting that the relative error is not too large.
           
        absolute_tolerance : float
        
            For small values, the absolute error will be checked against this tolerance,
            instead of considering the relative error.
        """
        assert(len(verified_values) == len(coordinates))
        
        for i, verified_value in enumerate(verified_values):
            
            values = model.solution.at(coordinates[i])
            
            value = values[component]
            
            if not(subcomponent == None):
            
                value = value[subcomponent]
            
            absolute_error = abs(value - verified_value)
            
            if abs(verified_value) > absolute_tolerance:
            
                relative_error = absolute_error/verified_value
           
                assert(relative_error < relative_tolerance)
                
            else:
            
                assert(absolute_error < absolute_tolerance)

                
def unsteadiness(model):
    
    return fe.norm(model.solution - model.initial_values[0], "L2")/ \
        fe.norm(model.initial_values[0], "L2")
            
            
def test__verify_against_heat_driven_cavity_benchmark():

    model = fempy.benchmarks.heat_driven_cavity.Model(meshsize = 40)
    
    model.solver.solve()
    
    """ Verify against the result published in @cite{wang2010comprehensive}. """
    Ra = model.rayleigh_number.__float__()
    
    Pr = model.prandtl_number.__float__()
    
    """ Verify coordinates (0.3499, 0.8499, 0.9999) instead of (0.35, 0.85, 1)
    because the Function evaluation fails arbitrarily at these points.
    See https://github.com/firedrakeproject/firedrake/issues/1340 """
    verify_scalar_solution_component(
        model,
        component = 1,
        subcomponent = 0,
        coordinates = [(0.5, y) 
            for y in (0., 0.15, 0.34999, 0.5, 0.65, 0.84999, 0.99999)],
        verified_values = [val*Ra**0.5/Pr
            for val in (0.0000, -0.0649, -0.0194, 0.0000, 
                        0.0194, 0.0649, 0.0000)],
        relative_tolerance = 1.e-2,
        absolute_tolerance = 1.e-2*0.0649*Ra**0.5/Pr)
    