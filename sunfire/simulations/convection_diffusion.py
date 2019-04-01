""" A convection-diffusion simulation class """
import firedrake as fe
import sunfire.simulation


dot, grad, div = fe.dot, fe.grad, fe.div
    
def variational_form_residual(sim, solution):
    
    u = solution
    
    v = fe.TestFunction(solution.function_space())
    
    a = sim.advection_velocity
    
    nu = sim.kinematic_viscosity
    
    dx = fe.dx(degree = sim.quadrature_degree)
    
    return (v*dot(a, grad(u)) + dot(grad(v), nu*grad(u)))*dx
    
    
def strong_residual(sim, solution):
    
    x = fe.SpatialCoordinate(sim.mesh)
    
    u = solution
    
    a = sim.advection_velocity
    
    nu = sim.kinematic_viscosity
    
    return dot(a, grad(u)) - div(nu*grad(u))


def element(cell, degree):

    return fe.FiniteElement("P", cell, degree)
    
    
class Simulation(sunfire.simulation.Simulation):
    
    def __init__(self, *args,
            mesh, element_degree, advection_velocity,
            **kwargs):
        
        self.kinematic_viscosity = fe.Constant(1.)
        
        self.advection_velocity = advection_velocity(mesh)
    
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            variational_form_residual = variational_form_residual,
            **kwargs)
        