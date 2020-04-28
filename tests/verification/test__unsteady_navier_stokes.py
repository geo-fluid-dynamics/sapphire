"""Verify accuracy of the unsteady Navier-Stokes solver.

The pressure accuracy is not verified.
On the other hand, the pressure accuracy in `test__navier_stokes.py`
(steady state) is verified.
This should be investigated further.
"""
import firedrake as fe 
import sapphire.mms
import sapphire.simulations.unsteady_navier_stokes as sim_module
import tests.validation.helpers


pi, sin, exp = fe.pi, fe.sin, fe.exp

def space_verification_solution(sim):
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    ihat, jhat = sim.unit_vectors()
    
    u = exp(0.5)*sin(2.*pi*x)*sin(pi*y)*ihat + \
        exp(0.5)*sin(pi*x)*sin(2.*pi*y)*jhat
    
    p = -sin(pi*x - pi/2.)*sin(2.*pi*y - pi/2.)
    
    mean_pressure = fe.assemble(p*fe.dx)
    
    p -= mean_pressure
    
    return u, p
    
    
def time_verification_solution(sim):
    
    t = sim.time
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    ihat, jhat = sim.unit_vectors()
    
    u = exp(2.*t)*\
        (sin(2.*pi*x)*sin(pi*y)*ihat + 
        sin(pi*x)*sin(2.*pi*y)*jhat)
    
    p = -sin(pi*x - pi/2.)*sin(2.*pi*y - pi/2.)
    
    mean_pressure = fe.assemble(p*fe.dx)
    
    p -= mean_pressure
    
    return u, p
    
    
def dirichlet_boundary_conditions(sim, manufactured_solution):
    """Apply velocity Dirichlet BC's on every boundary.
    
    Do not apply Dirichlet BC's on the pressure.
    """
    W = sim.solution_space
    
    u, _ = manufactured_solution
    
    return [fe.DirichletBC(W.sub(0), u, "on_boundary"),]
    
    
sim_kwargs = {
    "reynolds_number": 3.,
    "quadrature_degree": 4}
    
    
def test__verify_second_order_spatial_convergence_via_mms():
    
    sim_kwargs["element_degrees"] = (2, 1)
    
    sim_kwargs["timestep_size"] = 1./16.
    
    sim_kwargs["time_stencil_size"] = 3
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = space_verification_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        meshes = [fe.UnitSquareMesh(n, n) for n in (5, 10, 20, 40)],
        norms = ("H1", "L2"),
        expected_orders = (2, 2),
        decimal_places = 1,
        endtime = 1.)
    
    
def test__verify_first_order_temporal_convergence_via_mms():
    
    sim_kwargs["element_degrees"] = (3, 2)
    
    sim_kwargs["mesh"] = fe.UnitSquareMesh(24, 24)
    
    sim_kwargs["time_stencil_size"] = 2
    
    sapphire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = time_verification_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        norms = ("L2", None),
        expected_orders = (1, None),
        endtime = 1.,
        timestep_sizes = (0.2, 0.1, 0.05),
        decimal_places = 1)


class LidDrivenCavitySimulation(sim_module.Simulation):
    
    def dirichlet_boundary_conditions(self):
        
        W = self.solution_space
        
        return [
            fe.DirichletBC(W.sub(0), fe.Constant((0., 0.)), (1, 2, 3)),
            fe.DirichletBC(W.sub(0), fe.Constant((1., 0.)), 4)]
            

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
    endtime = 1.e12
    
    sim = LidDrivenCavitySimulation(
        reynolds_number = 100.,
        mesh = fe.UnitSquareMesh(50, 50),
        element_degrees = (2, 1),
        timestep_size = endtime)
    
    sim.states = sim.run(endtime = endtime)
    
    tests.validation.helpers.check_scalar_solution_component(
        solution = sim.solution,
        component = 0,
        subcomponent = 0,
        coordinates = [(0.5, y) 
            for y in (0.9766, 0.1016, 0.0547, 0.0000)],
        expected_values = (0.8412, -0.0643, -0.0372, 0.0000),
        absolute_tolerances = (0.0025, 0.0015, 0.001, 1.e-16))
        