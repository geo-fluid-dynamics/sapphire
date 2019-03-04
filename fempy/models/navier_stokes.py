""" A steady incompressible Navier-Stokes model class """
import firedrake as fe
import fempy.model

    
class Model(fempy.model.Model):
    
    def __init__(self, *args, mesh, element_degree, **kwargs):
        
        element = fe.MixedElement(
            fe.VectorElement(
                "P", mesh.ufl_cell(), element_degree + 1),
            fe.FiniteElement(
                "P", mesh.ufl_cell(), element_degree))
                
        super().__init__(*args, mesh, element, **kwargs)
        
    def init_weak_form_residual(self):

        inner, dot, grad, div, sym = \
            fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
        u, p = fe.split(self.solution)
        
        psi_u, psi_p = fe.TestFunctions(self.solution.function_space())
        
        mass = psi_p*div(u)
        
        momentum = dot(psi_u, grad(u)*u) - div(psi_u)*p + \
            2.*inner(sym(grad(psi_u)), sym(grad(u)))
        
        self.weak_form_residual = mass + momentum
    
    def strong_form_residual(self, solution):
    
        grad, dot, div, sym = fe.grad, fe.dot, fe.div, fe.sym
        
        u, p = solution
        
        r_u = grad(u)*u + grad(p) - 2.*div(sym(grad(u)))
        
        r_p = div(u)
        
        return r_u, r_p
        