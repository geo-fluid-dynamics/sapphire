"""Verify accuracy of the unsteady Navier-Stokes-Boussinesq solver."""
import firedrake as fe 
import sapphire.mms
from sapphire.simulations import \
    unsteady_navier_stokes_boussinesq as sim_module
import tests.validation.helpers


sin, pi = fe.sin, fe.pi

def space_verification_solution(sim):
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    u0 = sin(2.*pi*x)*sin(pi*y)
    
    u1 = sin(pi*x)*sin(2.*pi*y)
    
    ihat, jhat = sim.unit_vectors()
    
    u = (u0*ihat + u1*jhat)
    
    p = -0.5*sin(pi*x)*sin(pi*y)
    
    T = sin(2.*pi*x)*sin(pi*y)
    
    mean_pressure = fe.assemble(p*fe.dx)
    
    p -= mean_pressure
    
    return p, u, T
    

def time_verification_solution(sim):
    
    exp = fe.exp
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    t = sim.time
    
    u0 = sin(2.*pi*x)*sin(pi*y)
    
    u1 = sin(pi*x)*sin(2.*pi*y)
    
    ihat, jhat = sim.unit_vectors()
    
    u = exp(t)*(u0*ihat + u1*jhat)
    
    p = -0.5*sin(pi*x)*sin(pi*y)
    
    T = exp(t)*sin(2.*pi*x)*sin(pi*y)
    
    mean_pressure = fe.assemble(p*fe.dx)
    
    p -= mean_pressure
    
    return p, u, T
    
    
def dirichlet_boundary_conditions(sim, manufactured_solution):
    """Apply velocity and temperature Dirichlet BC's on every boundary.
    
    Do not apply Dirichlet BC's on the pressure.
    """
    W = sim.solution_space
    
    _, u, T = manufactured_solution
    
    return [
        fe.DirichletBC(W.sub(1), u, "on_boundary"),
        fe.DirichletBC(W.sub(2), T, "on_boundary")]


sim_kwargs = {
    "reynolds_number": 20.,
    "rayleigh_number": 1.e3,
    "prandtl_number": 0.71,
    "quadrature_degree": 4}
    
def test__verify_second_order_spatial_convergence_via_mms():
    
    sim_kwargs["element_degrees"] = (1, 2, 2)
    
    sim_kwargs["timestep_size"] = 1.
    
    sim_kwargs["time_stencil_size"] = 2
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = space_verification_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        meshes = [fe.UnitSquareMesh(n, n) for n in (8, 16, 32)],
        norms = ("L2", "H1", "H1"),
        expected_orders = (2, 2, 2),
        decimal_places = 1,
        endtime = 1.)
    
 
def test__verify_first_order_temporal_convergence_via_mms():
    
    sim_kwargs["mesh"] = fe.UnitSquareMesh(30, 30)
    
    sim_kwargs["element_degrees"] = (2, 3, 3)
    
    sapphire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = time_verification_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        endtime = 1.,
        timestep_sizes = (1./2., 1./4., 1./8., 1./16.),
        norms = (None, "L2", "L2"),
        expected_orders = (None, 1, 1),
        decimal_places = 1)


class HeatDrivenCavitySimulation(sim_module.Simulation):
    
    def dirichlet_boundary_conditions(self):
        
        W = self.solution_space
        
        return [
            fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
            fe.DirichletBC(W.sub(2), 0.5, 1),
            fe.DirichletBC(W.sub(2), -0.5, 2)]


def test__steady_state_heat_driven_cavity_benchmark():
    """ Verify against steady state heat-driven cavity benchmark.
    
    Comparing to data published in @cite{wang2010comprehensive}.
    """
    endtime = 1.e12
    
    Ra = 1.e6
    
    Pr = 0.71
    
    sim = HeatDrivenCavitySimulation(
        rayleigh_number = Ra,
        prandtl_number = Pr,
        element_degrees = (1, 2, 2),
        mesh = fe.UnitSquareMesh(40, 40),
        timestep_size = endtime)
    
    sim.states = sim.run(endtime = endtime)
    
    # Check coordinates (0.3499, 0.8499) instead of (0.35, 0.85)
    # because the Function evaluation fails at the exact coordinates.
    # See https://github.com/firedrakeproject/firedrake/issues/1340 
    tests.validation.helpers.check_scalar_solution_component(
        solution = sim.solution,
        component = 1,
        subcomponent = 0,
        coordinates = [(0.5, y) 
            for y in (0., 0.15, 0.34999, 0.5, 0.65, 0.84999)],
        expected_values = [val*Ra**0.5/Pr
            for val in (0.0000, -0.0649, -0.0194, 0.0000, 
                        0.0194, 0.0649)],
        absolute_tolerances = [val*Ra**0.5/Pr 
            for val in (1.e-12, 0.001, 0.001, 1.e-12, 0.001, 0.001)])
