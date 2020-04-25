"""Validate against steady state heat-driven water cavity benchmark."""
import matplotlib
matplotlib.use('Agg')  # Only use back-end to prevent displaying image
import matplotlib.pyplot
import firedrake as fe 
import sapphire.simulations.examples.heat_driven_cavity
import sapphire.simulations.examples.heat_driven_cavity_with_water
import tests.validation.helpers


def test__heat_driven_cavity_with_water(tmpdir):
    
    sim = sapphire.simulations.examples.\
        heat_driven_cavity_with_water.Simulation(
            element_degree = (1, 2, 2),
            mesh = fe.UnitSquareMesh(20, 20),
            output_directory_path = tmpdir)
    
    sim.solution = sim.solve_with_continuation_on_grashof_number()
    
    
    p, u, T = sim.solution.split()
    
    fe.tricontourf(T)
    
    fe.quiver(u)
    
    filepath = tmpdir+"/T_and_u.png"
    
    print("Writing plot to {}".format(filepath))
    
    matplotlib.pyplot.savefig(filepath)
    
    
    dot, grad = fe.dot, fe.grad
    
    coldwall_id = sapphire.simulations.examples.heat_driven_cavity.coldwall_id
    
    ds = fe.ds(subdomain_id=coldwall_id)
    
    nhat = fe.FacetNormal(sim.mesh)
    
    p, u, T = fe.split(sim.solution)
    
    coldwall_heatflux = fe.assemble(dot(grad(T), nhat)*ds)

    print("Integrated cold wall heat flux = {}".format(coldwall_heatflux))
    
    assert(round(coldwall_heatflux, 0) == -8.)
    