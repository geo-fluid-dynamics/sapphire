""" Verify accuracy of the Navier-Stokes solver. """
import firedrake as fe 
import sapphire.mms
from sapphire.simulations.navier_stokes import Simulation


grad, div, sym = fe.grad, fe.div, fe.sym
    
def strong_residual(sim, solution):
    
    p, u = solution
    
    Re = sim.reynolds_number
    
    r_p = div(u)
    
    r_u = grad(u)*u + grad(p) - 2./Re*div(sym(grad(u)))
    
    return r_p, r_u
    

def manufactured_solution(sim):
    
    sin, pi = fe.sin, fe.pi
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    ihat, jhat = sim.unit_vectors
    
    u = sin(2.*pi*x)*sin(pi*y)*ihat + \
        sin(pi*x)*sin(2.*pi*y)*jhat
    
    p = -sin(pi*x - pi/2.)*sin(2.*pi*y - pi/2.)
    
    mean_pressure = fe.assemble(p*fe.dx)
    
    p -= mean_pressure
    
    return p, u
    
    
class UnitSquareSimulation(Simulation):
    
    def __init__(self, *args,
            meshcell_size,
            **kwargs):
        
        n = int(round(1/meshcell_size))
        
        kwargs["mesh"] = fe.UnitSquareMesh(n, n)
        
        super().__init__(*args, **kwargs)
        
        
def dirichlet_boundary_conditions(sim, manufactured_solution):
    """Apply velocity Dirichlet BC's on every boundary."""
    
    p, u = manufactured_solution
    
    return [fe.DirichletBC(
        sim.solution_subspaces["u"],
        u,
        "on_boundary"),]
    
    
sim_kwargs = {"reynolds_number": 3.}
    
def test__verify_spatial_convergence__second_order__via_mms():
    
    sim_kwargs["taylor_hood_pressure_degree"] = 1
    
    def table_column_value_from_parameter_value(mesh):
        
        return mesh.cell_sizes((0.,)*mesh.geometric_dimension())
        
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = "meshcell_size",
        discretization_parameter_values = [1/n for n in (4, 8, 16, 32, 64)],
        Simulation = UnitSquareSimulation,
        sim_kwargs = sim_kwargs,
        strong_residual = strong_residual,
        manufactured_solution = manufactured_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        norms = ("L2", "H1"),
        points_in_rate_estimator = 3,
        expected_orders = (2, 2),
        decimal_places = 1)
        

def test__verify_spatial_convergence__third_order__via_mms():
    
    sim_kwargs["taylor_hood_pressure_degree"] = 2
    
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = "meshcell_size",
        discretization_parameter_values = [1/n for n in (4, 8, 16)],
        Simulation = UnitSquareSimulation,
        sim_kwargs = sim_kwargs,
        strong_residual = strong_residual,
        manufactured_solution = manufactured_solution,
        time_dependent = False,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        norms = ("L2", "H1"),
        points_in_rate_estimator = 3,
        expected_orders = (3, 3),
        decimal_places = 1)
        