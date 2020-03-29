import firedrake as fe 
import sapphire.simulations.navier_stokes


def test__lid_driven_cavity_benchmark():
    """ Validate against steady state lid-driven cavity benchmark.
    
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
    def initial_values(sim):

        return sim.solution
        
    def dirichlet_boundary_conditions(sim):
        
        W = sim.function_space
        
        return [
            fe.DirichletBC(W.sub(0), fe.Constant((0., 0.)), (1, 2, 3)),
            fe.DirichletBC(W.sub(0), fe.Constant((1., 0.)), 4)]
    
    sim = sapphire.simulations.navier_stokes.Simulation(
        reynolds_number = 100.,
        initial_values = initial_values,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        mesh = fe.UnitSquareMesh(50, 50),
        element_degree = (2, 1))
    
    sim.solution = sim.solve()
    
    sapphire.test.check_scalar_solution_component(
        solution = sim.solution,
        component = 0,
        subcomponent = 0,
        coordinates = [(0.5, y) 
            for y in (0.9766, 0.1016, 0.0547, 0.0000)],
        expected_values = (0.8412, -0.0643, -0.0372, 0.0000),
        absolute_tolerances = (0.0025, 0.0015, 0.001, 1.e-12))
        