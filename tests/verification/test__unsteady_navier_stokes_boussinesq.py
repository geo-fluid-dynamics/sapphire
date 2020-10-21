"""Verify accuracy of the unsteady Navier-Stokes-Boussinesq solver."""
import firedrake as fe 
import sapphire.mms
import tests.verification.test__navier_stokes_boussinesq
from sapphire.simulations.unsteady_navier_stokes_boussinesq import Simulation
import tests.validation.helpers


diff = fe.diff

def strong_residual(sim, solution):
    
    r_p, r_u, r_T = tests.verification.test__navier_stokes_boussinesq.\
        strong_residual(sim = sim, solution = solution)
    
    _, u, T = solution
    
    t = sim.time
    
    r_u += diff(u, t)
    
    r_T += diff(T, t)
    
    return r_p, r_u, r_T
    

sin, pi = fe.sin, fe.pi

def space_verification_solution(sim):
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    u0 = sin(2*pi*x)*sin(pi*y)
    
    u1 = sin(pi*x)*sin(2*pi*y)
    
    ihat, jhat = sim.unit_vectors
    
    u = (u0*ihat + u1*jhat)
    
    p = -0.5*sin(pi*x)*sin(pi*y)
    
    T = sin(2*pi*x)*sin(pi*y)
    
    mean_pressure = fe.assemble(p*fe.dx)
    
    p -= mean_pressure
    
    return p, u, T
    

def time_verification_solution(sim):
    
    exp = fe.exp
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    t = sim.time
    
    u0 = sin(2*pi*x)*sin(pi*y)
    
    u1 = sin(pi*x)*sin(2*pi*y)
    
    ihat, jhat = sim.unit_vectors
    
    u = exp(t)*(u0*ihat + u1*jhat)
    
    p = -0.5*sin(pi*x)*sin(pi*y)
    
    T = exp(t)*sin(2*pi*x)*sin(pi*y)
    
    mean_pressure = fe.assemble(p*fe.dx)
    
    p -= mean_pressure
    
    return p, u, T
    
    
class UnitSquareUniformMeshSimulation(Simulation):
    
    def __init__(self, *args,
            meshcell_size,
            **kwargs):
        
        n = int(round(1/meshcell_size))
        
        kwargs["mesh"] = fe.UnitSquareMesh(n, n)
        
        super().__init__(*args, **kwargs)
    
    
def dirichlet_boundary_conditions(sim, manufactured_solution):
    """Apply velocity and temperature Dirichlet BC's on every boundary.
    
    Do not apply Dirichlet BC's on the pressure.
    """
    _, u, T = manufactured_solution
    
    return [
        fe.DirichletBC(sim.solution_subspaces["u"], u, "on_boundary"),
        fe.DirichletBC(sim.solution_subspaces["T"], T, "on_boundary")]


sim_kwargs = {
    "reynolds_number": 20,
    "rayleigh_number": 1.e3,
    "prandtl_number": 0.71,
    "quadrature_degree": 4}
    
def test__verify_second_order_spatial_convergence_via_mms():
    
    sim_kwargs["taylor_hood_pressure_degree"] = 1
    
    sim_kwargs["temperature_degree"] = 2
    
    sim_kwargs["timestep_size"] = 1
    
    sim_kwargs["time_stencil_size"] = 2
    
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = "meshcell_size",
        discretization_parameter_values = [1/n for n in (8, 16, 32)],
        Simulation = UnitSquareUniformMeshSimulation,
        sim_kwargs = sim_kwargs,
        strong_residual = strong_residual,
        manufactured_solution = space_verification_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        norms = ("L2", "H1", "H1"),
        expected_orders = (2, 2, 2),
        decimal_places = 1,
        endtime = 1)
    
 
def test__verify_first_order_temporal_convergence_via_mms():
    
    sim_kwargs["meshcell_size"] = 1/32
    
    sim_kwargs["taylor_hood_pressure_degree"] = 2
    
    sim_kwargs["temperature_degree"] = 3
    
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = "timestep_size",
        discretization_parameter_values = (1/2, 1/4, 1/8, 1/16),
        Simulation = UnitSquareUniformMeshSimulation,
        sim_kwargs = sim_kwargs,
        strong_residual = strong_residual,
        manufactured_solution = time_verification_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        endtime = 1,
        norms = (None, "L2", "L2"),
        expected_orders = (None, 1, 1),
        decimal_places = 1)


class HeatDrivenCavitySimulation(UnitSquareUniformMeshSimulation):
    
    def dirichlet_boundary_conditions(self):
        
        return [
            fe.DirichletBC(self.solution_subspaces["u"], (0, 0), "on_boundary"),
            fe.DirichletBC(self.solution_subspaces["T"], 0.5, 1),
            fe.DirichletBC(self.solution_subspaces["T"], -0.5, 2)]


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
        taylor_hood_pressure_degree = 1,
        temperature_degree = 2,
        meshcell_size = 1/40,
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
            for y in (0, 0.15, 0.34999, 0.5, 0.65, 0.84999)],
        expected_values = [val*Ra**0.5/Pr
            for val in (0, -0.0649, -0.0194, 0, 
                        0.0194, 0.0649)],
        absolute_tolerances = [val*Ra**0.5/Pr 
            for val in (1.e-12, 0.001, 0.001, 1.e-12, 0.001, 0.001)])
