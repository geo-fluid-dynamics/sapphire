"""Verify accuracy of the unsteady Navier-Stokes-Boussinesq solver."""
import firedrake as fe 
import sapphire.mms
from sapphire.simulation import unit_vectors
from sapphire.simulations import \
    unsteady_navier_stokes_boussinesq as sim_module
import tests.validation.helpers


def space_verification_solution(sim):
    
    sin, pi = fe.sin, fe.pi
    
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
    
    exp, sin, pi = fe.exp, fe.sin, fe.pi
    
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
    

sim_kwargs = {
    "grashof_number": 2.,
    "prandtl_number": 0.71,
    "quadrature_degree": None}
    
def test__verify_spatial_convergence__second_order__via_mms():
    
    sim_kwargs["timestep_size"] = 1.
    
    sim_kwargs["element_degree"] = (1, 2, 2)
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = space_verification_solution,
        endtime = 1.,
        meshes = [fe.UnitSquareMesh(n, n) for n in (4, 8, 16, 32, 64)],
        norms = ("L2", "H1", "H1"),
        expected_orders = (2, 2, 2),
        decimal_places = 1)
    
 
def test__verify_temporal_convergence__first_order__via_mms():
    
    meshsize = 48
    
    sim_kwargs["mesh"] = fe.UnitSquareMesh(meshsize, meshsize)
    
    sim_kwargs["element_degree"] = (2, 3, 3)
    
    sim_kwargs["time_stencil_size"] = 2
    
    sapphire.mms.verify_temporal_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = time_verification_solution,
        endtime = 1.,
        timestep_sizes = (1./2., 1./4., 1./8., 1./16.),
        norms = (None, "L2", "L2"),
        expected_orders = (None, 1, 1),
        decimal_places = 1)
        
        
def test__steady_state_heat_driven_cavity():
    """ Verify against steady state heat-driven cavity benchmark.
    
    Comparing to data published in @cite{wang2010comprehensive}.
    """
    Ra = 1.e6

    Pr = 0.71

    Gr = Ra/Pr

    def initial_values(sim):

        return sim.solution
        
    def dirichlet_boundary_conditions(sim):
        
        W = sim.function_space
        
        return [
            fe.DirichletBC(W.sub(1), (0., 0.), "on_boundary"),
            fe.DirichletBC(W.sub(2), 0.5, 1),
            fe.DirichletBC(W.sub(2), -0.5, 2)]
    
    meshsize = 40
    
    endtime = 1.e12
    
    sim = sapphire.simulations.unsteady_navier_stokes_boussinesq.Simulation(
        prandtl_number = Pr,
        grashof_number = Gr,
        initial_values = initial_values,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        mesh = fe.UnitSquareMesh(meshsize, meshsize),
        element_degree = (1, 2, 2),
        timestep_size = endtime)
    
    sim.solutions, _ = sim.run(endtime = endtime)
    
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
            