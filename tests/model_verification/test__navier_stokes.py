import firedrake as fe 
import sapphire.mms
import sapphire.simulations.navier_stokes as sim_module


def manufactured_solution(sim):
    
    sin, pi = fe.sin, fe.pi
    
    x = fe.SpatialCoordinate(sim.mesh)
    
    ihat, jhat = sapphire.simulation.unit_vectors(sim.mesh)
    
    u = sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
        sin(pi*x[0])*sin(2.*pi*x[1])*jhat
    
    p = -0.5*(u[0]**2 + u[1]**2)
    
    return u, p
    
    
def test__verify_convergence_order_via_mms(
        mesh_sizes = (8, 16, 32), tolerance = 0.2):
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = {"reynolds_number": 3.},
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitSquareMesh(n, n) for n in mesh_sizes],
        norms = ("H1", "L2"),
        expected_orders = (2, 2),
        tolerance = tolerance)
        
        
def test__lid_driven_cavity_benchmark():
    """ Validate against steady state lid-driven cavity benchmark.
    
    Comparing to data published in 
    
        @article{ghia1982high-re,
            author = {Ghia, Urmila and Ghia, K.N and Shin, C.T},
            year = {1982},
            month = {12},
            pages = {387-411},
            title = {High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method1},
            volume = {48},
            journal = {Journal of Computational Physics},
            doi = {10.1016/0021-9991(82)90058-4}
        }
    """
    def initial_values(sim):

        return sim.solution
        
    def dirichlet_boundary_conditions(sim):
        
        W = sim.function_space
        
        return [
            fe.DirichletBC(W.sub(0), fe.Constant((0., 0.)), (1, 2, 3)),
            fe.DirichletBC(W.sub(0), fe.Constant((1., 0.)), 4)]
    
    sim = sapphire.simulations.navier_stokes.Simulation(
        reynolds_number = 100.,
        initial_values = initial_values,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        mesh = fe.UnitSquareMesh(20, 20),
        element_degree = (2, 1))
    
    sim.solution = sim.solve()
    
    sapphire.test.check_scalar_solution_component(
        solution = sim.solution,
        component = 0,
        subcomponent = 0,
        coordinates = [(0.5, y) 
            for y in (0.9999, 0.9766, 0.1016, 0.0547, 0.0000)],
            # Check y = 0.9999 instead of y = 1
            # because the Function evaluation fails at the exact coordinate.
            # See https://github.com/firedrakeproject/firedrake/issues/1340 
        expected_values = (1.0000, 0.8412, -0.0643, -0.0372, 0.0000),
        relative_tolerance = 1.e-2,
        absolute_tolerance = 1.e-2)
        