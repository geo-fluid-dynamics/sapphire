""" A convection-diffusion simulation class """
import firedrake as fe
import sapphire.simulation


dot, grad, div = fe.dot, fe.grad, fe.div
    
def variational_form_residual(sim, solution):
    
    u = solution
    
    v = fe.TestFunction(solution.function_space())
    
    a = sim.advection_velocity
    
    d = sim.diffusion_coefficient
    
    dx = fe.dx(degree = sim.quadrature_degree)
    
    return (v*dot(a, grad(u)) + dot(grad(v), d*grad(u)))*dx
    
    
def strong_residual(sim, solution):
    
    x = fe.SpatialCoordinate(sim.mesh)
    
    u = solution
    
    a = sim.advection_velocity
    
    d = sim.diffusion_coefficient
    
    return dot(a, grad(u)) - d*div(grad(u))


def element(cell, degree):

    return fe.FiniteElement("P", cell, degree)
    
    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args,
            mesh, advection_velocity, element_degree = 1,
            **kwargs):
        
        self.diffusion_coefficient = fe.Constant(1.)
        
        self.advection_velocity = advection_velocity(mesh)
    
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            variational_form_residual = variational_form_residual,
            **kwargs)
        