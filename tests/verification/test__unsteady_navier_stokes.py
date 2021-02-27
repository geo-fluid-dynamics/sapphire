"""Verify accuracy of the unsteady Navier-Stokes solver."""
import firedrake as fe 
import sapphire.mms
import tests.verification.test__navier_stokes
from sapphire.simulations.unsteady_navier_stokes import Simulation
import tests.validation.helpers


diff = fe.diff

def strong_residual(sim, solution):
    
    r_p, r_u = tests.verification.test__navier_stokes.\
        strong_residual(sim = sim, solution = solution)
    
    _, u = solution
    
    t = sim.time
    
    r_u += diff(u, t)
    
    return r_p, r_u

    
pi, sin, exp = fe.pi, fe.sin, fe.exp

def space_verification_solution(sim):
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    p = -sin(pi*x - pi/2.)*sin(2.*pi*y - pi/2.)
    
    mean_pressure = fe.assemble(p*fe.dx)
    
    p -= mean_pressure
    
    ihat, jhat = sim.unit_vectors
    
    u = exp(0.5)*sin(2.*pi*x)*sin(pi*y)*ihat + \
        exp(0.5)*sin(pi*x)*sin(2.*pi*y)*jhat
    
    return p, u
    
    
def time_verification_solution(sim):
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    p = -sin(pi*x - pi/2.)*sin(2.*pi*y - pi/2.)
    
    mean_pressure = fe.assemble(p*fe.dx)
    
    p -= mean_pressure
    
    t = sim.time
    
    ihat, jhat = sim.unit_vectors
    
    u = exp(2.*t)*\
        (sin(2.*pi*x)*sin(pi*y)*ihat + 
        sin(pi*x)*sin(2.*pi*y)*jhat)
    
    return p, u


class UnitSquareUniformMeshSimulation(Simulation):
    
    def __init__(self, *args,
            meshcell_size,
            **kwargs):
        
        n = int(round(1/meshcell_size))
        
        kwargs["mesh"] = fe.UnitSquareMesh(n, n)
        
        super().__init__(*args, **kwargs)
        
    
def dirichlet_boundary_conditions(sim, manufactured_solution):
    """Apply velocity Dirichlet BC's on every boundary.
    
    Do not apply Dirichlet BC's on the pressure.
    """
    _, u = manufactured_solution
    
    return [fe.DirichletBC(sim.solution_subspaces["u"], u, "on_boundary"),]
    
    
sim_kwargs = {
    "reynolds_number": 3.,
    "quadrature_degree": 4}

def test__verify_second_order_spatial_convergence_via_mms(tmpdir):
    
    sim_kwargs["taylor_hood_pressure_degree"] = 1
    
    sim_kwargs["timestep_size"] = 1./16.
    
    sim_kwargs["time_stencil_size"] = 3
    
    with open(tmpdir + "/convergence_table.csv", "w") as outfile:
    
        sapphire.mms.verify_order_of_accuracy(
            discretization_parameter_name = "meshcell_size",
            discretization_parameter_values = [1/n for n in (5, 10, 20, 40)],
            Simulation = UnitSquareUniformMeshSimulation,
            sim_kwargs = sim_kwargs,
            strong_residual = strong_residual,
            manufactured_solution = space_verification_solution,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            norms = ("L2", "H1"),
            expected_orders = (2, 2),
            decimal_places = 1,
            endtime = 1.,
            outfile = outfile)
        
    
def test__verify_first_order_temporal_convergence_via_mms(tmpdir):
    
    sim_kwargs["taylor_hood_pressure_degree"] = 2
    
    sim_kwargs["meshcell_size"] = 1/24
    
    sim_kwargs["time_stencil_size"] = 2
    
    with open(tmpdir + "/convergence_table.csv", "w") as outfile:
        
        sapphire.mms.verify_order_of_accuracy(
            discretization_parameter_name = "timestep_size",
            discretization_parameter_values = (0.2, 0.1, 0.05),
            Simulation = UnitSquareUniformMeshSimulation,
            sim_kwargs = sim_kwargs,
            strong_residual = strong_residual,
            manufactured_solution = time_verification_solution,
            dirichlet_boundary_conditions = dirichlet_boundary_conditions,
            norms = (None, "L2"),
            expected_orders = (None, 1),
            endtime = 1.,
            decimal_places = 1,
            outfile = outfile)


class LidDrivenCavitySimulation(UnitSquareUniformMeshSimulation):
    
    def dirichlet_boundary_conditions(self):
        
        return [
            fe.DirichletBC(
                self.solution_subspaces["u"],
                fe.Constant((0., 0.)),
                (1, 2, 3)),
            fe.DirichletBC(
                self.solution_subspaces["u"],
                fe.Constant((1., 0.)),
                4)]
            

def test__steady_state_lid_driven_cavity_benchmark():
    """ Verify against steady state lid-driven cavity benchmark.
    
    Comparing to data published in 
    
        @article{ghia1982high-re,
            author = {Ghia, Urmila and Ghia, K.N and Shin, C.T},
            year = {1982},
            month = {12},
            pages = {387-411},
            title = {High-Re solutions for incompressible flow using
                the Navier-Stokes equations and a multigrid method1},
            volume = {48},
            journal = {Journal of Computational Physics},
            doi = {10.1016/0021-9991(82)90058-4}
        }
    """
    endtime = 1.e12
    
    sim = LidDrivenCavitySimulation(
        reynolds_number = 100.,
        meshcell_size = 1/50,
        taylor_hood_pressure_degree = 1,
        timestep_size = endtime)
    
    sim.states = sim.run(endtime = endtime)
    
    tests.validation.helpers.check_scalar_solution_component(
        solution = sim.solution,
        component = sim.fieldnames.index("u"),
        subcomponent = 0,
        coordinates = [(0.5, y) 
            for y in (0.9766, 0.1016, 0.0547, 0.0000)],
        expected_values = (0.8412, -0.0643, -0.0372, 0.0000),
        absolute_tolerances = (0.0025, 0.0015, 0.001, 1.e-16))
        
