""" A steady incompressible Navier-Stokes simulation class """
import firedrake as fe
import sapphire.simulation


inner, dot, grad, div, sym = \
        fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
def weak_form_residual(sim, solution):
    
    u, p = fe.split(solution)
    
    psi_u, psi_p = fe.TestFunctions(solution.function_space())
    
    Re = sim.reynolds_number
    
    mass = psi_p*div(u)
    
    momentum = dot(psi_u, grad(u)*u) - div(psi_u)*p + \
        2./Re*inner(sym(grad(psi_u)), sym(grad(u)))
    
    dx = fe.dx(degree = sim.quadrature_degree)
    
    return (mass + momentum)*dx
    
    
def element(cell, degree):

    return fe.MixedElement(
        fe.VectorElement("P", cell, degree[0]),
        fe.FiniteElement("P", cell, degree[1]))
        
        
def strong_residual(sim, solution):
    
    u, p = solution
    
    Re = sim.reynolds_number
    
    r_u = grad(u)*u + grad(p) - 2./Re*div(sym(grad(u)))
    
    r_p = div(u)
    
    return r_u, r_p
    
    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args, 
            mesh,
            element_degree = (2, 1),
            reynolds_number = 1.,
            **kwargs):
        
        self.reynolds_number = fe.Constant(reynolds_number)
        
        super().__init__(*args,
            mesh = mesh,
            element = element(
                cell = mesh.ufl_cell(), degree = element_degree),
            weak_form_residual = weak_form_residual,
            time_stencil_size = 1,
            **kwargs)
        