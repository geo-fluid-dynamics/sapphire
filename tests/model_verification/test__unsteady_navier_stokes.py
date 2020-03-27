import firedrake as fe 
import sapphire.mms
from sapphire.simulation import unit_vectors
from sapphire.simulations import unsteady_navier_stokes as sim_module


def manufactured_solution(sim):
    
    exp, sin, pi = fe.exp, fe.sin, fe.pi
    
    x = fe.SpatialCoordinate(sim.mesh)
    
    t = sim.time
    
    ihat, jhat = unit_vectors(sim.mesh)
    
    u = exp(t)*(sin(2.*pi*x[0])*sin(pi*x[1])*ihat + \
        sin(pi*x[0])*sin(2.*pi*x[1])*jhat)
    
    p = -0.5*(u[0]**2 + u[1]**2)
    
    return u, p
    
    
sim_kwargs = {
    "quadrature_degree": 4,
    "element_degree": (2, 1),
    "time_stencil_size": 2}
    
def test__verify_spatial_convergence__second_order__via_mms():
    
    sim_kwargs["timestep_size"] = 1./32.
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitSquareMesh(n, n) for n in (3, 6, 12, 24)],
        norms = ("H1", "L2"),
        expected_orders = (2, 2),
        tolerance = 0.3,
        endtime = 1.)
    
 
def test__verify_temporal_convergence__first_order__via_mms():
    
    sim_kwargs["mesh"] = fe.UnitSquareMesh(32, 32)
    
    sapphire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = manufactured_solution,
        norms = (None, "L2"),
        expected_orders = (None, 1),
        endtime = 1.,
        timestep_sizes = (1./2., 1./4., 1./8.),
        tolerance = 0.1)
        
 
def test__steady_state_lid_driven_cavity_benchmark():
    """ Verify against steady state lid-driven cavity benchmark.
    
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
    
    endtime = 1.e12
    
    sim = sapphire.simulations.unsteady_navier_stokes.Simulation(
        reynolds_number = 100.,
        initial_values = initial_values,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        mesh = fe.UnitSquareMesh(50, 50),
        element_degree = (2, 1),
        timestep_size = endtime)
    
    sim.solutions, _ = sim.run(endtime = endtime)
    
    sapphire.test.check_scalar_solution_component(
        solution = sim.solution,
        component = 0,
        subcomponent = 0,
        coordinates = [(0.5, y) 
            for y in (0.9766, 0.1016, 0.0547, 0.0000)],
        expected_values = (0.8412, -0.0643, -0.0372, 0.0000),
        absolute_tolerances = (0.0025, 0.0015, 0.001, 1.e-16))
        