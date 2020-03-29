import firedrake as fe 
import sapphire.mms
import sapphire.simulations.navier_stokes as sim_module


def manufactured_solution(sim):
    
    sin, pi = fe.sin, fe.pi
    
    
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    
    
    ihat, jhat = sim.unit_vectors()
    
    u = sin(2.*pi*x)*sin(pi*y)*ihat + \
        sin(pi*x)*sin(2.*pi*y)*jhat
    
    p = -0.5*(u[0]**2 + u[1]**2)
    
    return u, p
    
    
def dirichlet_boundary_conditions(sim, manufactured_solution):
    """Apply velocity Dirichlet BC's on every boundary."""
    W = sim.function_space
    
    u, p = manufactured_solution
    
    return [fe.DirichletBC(W.sub(0), u, "on_boundary"),]
    
    
def nullspace(sim):
    
    W = sim.function_space
    
    return fe.MixedVectorSpaceBasis(
        W, [W.sub(0), fe.VectorSpaceBasis(constant=True)])
    

sim_kwargs = {
    "reynolds_number": 3.,
    "quadrature_degree": None,
    "element_degree": (2, 1),
    
    "nullspace": nullspace}
    
def test__verify_spatial_convergence__second_order__via_mms():
    
    
    
    sapphire.mms.verify_spatial_order_of_accuracy(
        sim_module = sim_module,
        sim_kwargs = sim_kwargs,
        manufactured_solution = manufactured_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        meshes = [fe.UnitSquareMesh(n, n) for n in (8, 16, 32, 64)],
        norms = ("H1", "L2"),
        expected_orders = (2, 2),
        decimal_places = 1)

