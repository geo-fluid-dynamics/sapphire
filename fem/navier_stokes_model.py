""" Implement a Poisson solver with Firedrake and verify it via MMS. """
import firedrake as fe
import fem.abstract_model

    
class NavierStokesModel(fem.abstract_model.AbstractModel):
    
    def element(self):
    
        return fe.MixedElement(
            fe.VectorElement('P', self.mesh.ufl_cell(), 2),
            fe.FiniteElement('P', self.mesh.ufl_cell(), 1))
    
    def weak_form_residual(self):

        inner, dot, grad, div, sym = \
            fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
        u, p = fe.split(self.solution)
        
        psi_u, psi_p = fe.TestFunctions(self.solution.function_space())
        
        dx = fe.dx(degree = 3)
        
        return (dot(psi_u, dot(grad(u), u))
            - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), sym(grad(u)))
            + psi_p*(div(u)))*dx
    