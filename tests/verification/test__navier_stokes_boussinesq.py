"""Verify accuracy of the Navier-Stokes-Boussinesq solver."""
import firedrake as fe 
import sapphire.mms
from sapphire.simulations.navier_stokes_boussinesq import Simulation


dot, grad, div, sym = fe.dot, fe.grad, fe.div, fe.sym
    
def strong_residual(sim, solution):
    
    Re = sim.reynolds_number
    
    Pr = sim.prandtl_number
    
    p, u, T = solution
    
    b = sim.buoyancy(temperature = T)
    
    r_p = div(u)
    
    r_u = grad(u)*u + grad(p) - 2./Re*div(sym(grad(u))) + b
    
    r_T = dot(u, grad(T)) - 1./(Re*Pr)*div(grad(T))
    
    return r_p, r_u, r_T
    

def manufactured_solution(sim):
    
    sin, pi = fe.sin, fe.pi
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    u0 = sin(2.*pi*x)*sin(pi*y)
    
    u1 = sin(pi*x)*sin(2.*pi*y)
    
    ihat, jhat = sim.unit_vectors
    
    u = u0*ihat + u1*jhat
    
    p = -0.5*(u0**2 + u1**2)
    
    T = sin(2.*pi*x)*sin(pi*y)
    
    mean_pressure = fe.assemble(p*fe.dx)
    
    p -= mean_pressure
    
    return p, u, T
    
    
class UnitSquareSimulation(Simulation):
    
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


def test__verify_second_order_spatial_accuracy_via_mms():
    
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = "meshcell_size",
        discretization_parameter_values = [1/n for n in (4, 8, 16, 32, 64)],
        Simulation = UnitSquareSimulation,
        sim_kwargs = {
            "reynolds_number": 20.,
            "rayleigh_number": 10.,
            "prandtl_number": 0.7,
            "taylor_hood_pressure_degree": 1,
            "temperature_degree": 2,
            "quadrature_degree": 4},
        strong_residual = strong_residual,
        manufactured_solution = manufactured_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        norms = ("L2", "H1", "H1"),
        expected_orders = (2, 2, 2),
        decimal_places = 1)
