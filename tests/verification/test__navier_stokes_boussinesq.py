"""Verify accuracy of the Navier-Stokes-Boussinesq solver."""
import firedrake as fe 
import sapphire.mms
import sapphire.simulations.navier_stokes_boussinesq as sim_module


def manufactured_solution(sim):
    
    sin, pi = fe.sin, fe.pi
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    u0 = sin(2.*pi*x)*sin(pi*y)
    
    u1 = sin(pi*x)*sin(2.*pi*y)
    
    ihat, jhat = sim.unit_vectors()
    
    u = u0*ihat + u1*jhat
    
    p = -0.5*(u0**2 + u1**2)
    
    T = sin(2.*pi*x)*sin(pi*y)
    
    mean_pressure = fe.assemble(p*fe.dx)
    
    p -= mean_pressure
    
    return p, u, T
    
    
def test__verify_spatial_accuracy_via_mms():
    
    Ra = 10.
    
    Pr = 0.7
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = {
            "grashof_number": Ra/Pr,
            "prandtl_number": Pr,
            "element_degree": (1, 2, 2),
            },
        manufactured_solution = manufactured_solution,
        meshes = [fe.UnitSquareMesh(n, n) for n in (4, 8, 16, 32, 64)],
        norms = ("L2", "H1", "H1"),
        expected_orders = (2, 2, 2),
        decimal_places = 1)
