import firedrake as fe 
import sapphire.mms
import sapphire.simulations.navier_stokes_boussinesq as sim_module
import sapphire.benchmarks.heat_driven_cavity


def manufactured_solution(sim):
    
    sin, pi = fe.sin, fe.pi
    
    x = fe.SpatialCoordinate(sim.mesh)
    
    u0 = sin(2.*pi*x[0])*sin(pi*x[1])
    
    u1 = sin(pi*x[0])*sin(2.*pi*x[1])
    
    ihat, jhat = sim.unit_vectors()
    
    u = u0*ihat + u1*jhat
    
    p = -0.5*(u0**2 + u1**2)
    
    T = sin(2.*pi*x[0])*sin(pi*x[1])
    
    return p, u, T
    
    
def test__verify_taylor_hood_accuracy_via_mms(
        mesh_sizes = (4, 8, 16, 32), 
        tolerance = 0.1):
    
    Ra = 10.
    
    Pr = 0.7
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_parameters = {
            "grashof_number": Ra/Pr,
            "prandtl_number": Pr,
            "element_degree": (1, 2, 2),
            },
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
        norms = ("L2", "H1", "H1"),
        expected_orders = (2, 2, 2),
        tolerance = tolerance)
    
    
def test__show_equal_order_pressure_penalty_inaccuracy_via_mms(
        mesh_sizes = (4, 8, 16, 32, 64), 
        tolerance = 0.1):
    
    Ra = 10.
    
    Pr = 0.7
    
    gamma = 1.e-7
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_parameters = {
            "grashof_number": Ra/Pr, 
            "prandtl_number": Pr, 
            "pressure_penalty_constant": gamma,
            "element_degree": (1, 1, 1),
            },
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
        norms = ("L2", "H1", "H1"),
        expected_orders = (None, 1, 1),
        tolerance = tolerance)
    
    
def test__verify_taylor_hood_with_pressure_penalty_accuracy_via_mms(
        mesh_sizes = (4, 8, 16, 32), 
        tolerance = 0.1):
    
    Ra = 10.
    
    Pr = 0.7
    
    gamma = 1.e-7
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_parameters = {
            "grashof_number": Ra/Pr, 
            "prandtl_number": Pr, 
            "pressure_penalty_constant": gamma,
            "element_degree": (1, 2, 2),
            },
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
        norms = ("L2", "H1", "H1"),
        expected_orders = (2, 2, 2),
        tolerance = tolerance)
    
    
def verify_scalar_solution_component(
            sim, 
            component, 
            coordinates, 
            verified_values, 
            relative_tolerance, 
            absolute_tolerance,
            subcomponent = None):
        """ Verify the scalar values of a specified solution component.
        
        Parameters
        ----------
        sim : sapphire.Simulation
        
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
            
            values = sim.solution.at(coordinates[i])
            
            value = values[component]
            
            if not(subcomponent == None):
            
                value = value[subcomponent]
            
            absolute_error = abs(value - verified_value)
            
            if abs(verified_value) > absolute_tolerance:
            
                relative_error = absolute_error/verified_value
           
                assert(relative_error < relative_tolerance)
                
            else:
            
                assert(absolute_error < absolute_tolerance)

                
def test__verify_against_heat_driven_cavity_benchmark():

    sim = sapphire.benchmarks.heat_driven_cavity.Simulation(
        element_degree = (1, 2, 2), meshsize = 40)
    
    sim.solution = sim.solve()
    
    """ Verify against the result published in @cite{wang2010comprehensive}. """
    Gr = sim.grashof_number.__float__()
    
    Pr = sim.prandtl_number.__float__()
    
    Ra = Gr*Pr
    
    """ Verify coordinates (0.3499, 0.8499, 0.9999) instead of (0.35, 0.85, 1)
    because the Function evaluation fails arbitrarily at these points.
    See https://github.com/firedrakeproject/firedrake/issues/1340 """
    verify_scalar_solution_component(
        sim,
        component = 1,
        subcomponent = 0,
        coordinates = [(0.5, y) 
            for y in (0., 0.15, 0.34999, 0.5, 0.65, 0.84999, 0.99999)],
        verified_values = [val*Ra**0.5/Pr
            for val in (0.0000, -0.0649, -0.0194, 0.0000, 
                        0.0194, 0.0649, 0.0000)],
        relative_tolerance = 1.e-2,
        absolute_tolerance = 1.e-2*0.0649*Ra**0.5/Pr)
    