""" A steady incompressible Navier-Stokes model class """
import firedrake as fe
import fem.model

    
class Model(fem.model.Model):
    
    def __init__(self):
    
        super().__init__()
        
    def init_element(self):
    
        self.element = fe.MixedElement(
            fe.VectorElement('P', self.mesh.ufl_cell(), 2),
            fe.FiniteElement('P', self.mesh.ufl_cell(), 1))
    
    def init_weak_form_residual(self):

        inner, dot, grad, div, sym = \
            fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
        u, p = fe.split(self.solution)
        
        psi_u, psi_p = fe.TestFunctions(self.solution.function_space())
        
        mass = psi_p*div(u)
        
        momentum = dot(psi_u, dot(grad(u), u)) - div(psi_u)*p + \
            2.*inner(sym(grad(psi_u)), sym(grad(u)))
        
        self.weak_form_residual = mass + momentum
    