"""Verify accuracy of the unsteady Navier-Stokes solver.

The pressure accuracy is not verified.
On the other hand, the pressure accuracy in `test__navier_stokes.py`
(steady state) is verified.
This should be investigated further.
"""
import firedrake as fe 
import sapphire.mms
import sapphire.simulations.unsteady_navier_stokes as sim_module


def manufactured_solution(sim):
    
    sin, pi = fe.sin, fe.pi
    
    exp = fe.exp
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    t = sim.time
    
    ihat, jhat = sim.unit_vectors()
    
    u = (sin(2.*pi*x)*sin(pi*y)*ihat + \
         sin(pi*x)*sin(2.*pi*y)*jhat)*exp(1. - t)
    
    p = -0.5*(u[0]**2 + u[1]**2)
    
    return u, p
    
    
def dirichlet_boundary_conditions(sim, manufactured_solution):
    """Apply velocity Dirichlet BC's on every boundary.
    
    Do not apply Dirichlet BC's on the pressure.
    """
    W = sim.function_space
    
    u, p = manufactured_solution
    
    return [fe.DirichletBC(W.sub(0), u, "on_boundary"),]
    
    
sim_kwargs = {
    "reynolds_number": 3.,
    "quadrature_degree": None}
    
    
def test__verify_spatial_convergence__second_order__via_mms():
    
    sim_kwargs["element_degree"] = (2, 1)
    
    sim_kwargs["timestep_size"] = 1./4.
    
    sim_kwargs["time_stencil_size"] = 3
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = manufactured_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        meshes = [fe.UnitSquareMesh(n, n) for n in (8, 16, 32)],
        norms = ("H1", None),
        expected_orders = (2, None),
        decimal_places = 1,
        endtime = 1.)
    
    
def test__verify_spatial_convergence__third_order__via_mms():
    
    sim_kwargs["element_degree"] = (3, 2)
    
    sim_kwargs["timestep_size"] = 1./32.
    
    sim_kwargs["time_stencil_size"] = 3
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = manufactured_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        meshes = [fe.UnitSquareMesh(n, n) for n in (4, 8, 16)],
        norms = ("H1", None),
        expected_orders = (3, None),
        decimal_places = 1,
        endtime = 1.)
    
 
def test__verify_temporal_convergence__second_order__via_mms():
    
    sim_kwargs["element_degree"] = (3, 2)
    
    sim_kwargs["mesh"] = fe.UnitSquareMesh(64, 64)
    
    sim_kwargs["time_stencil_size"] = 3
    
    sapphire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = manufactured_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        norms = ("L2", None),
        expected_orders = (2, None),
        endtime = 1.,
        timestep_sizes = (1./4., 1./8., 1./16., 1./32.),
        decimal_places = 1)
        
 
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
        