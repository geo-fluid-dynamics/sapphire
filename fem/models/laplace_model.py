""" Implement a Poisson solver with Firedrake and verify it via MMS. """
import firedrake as fe
import fem.abstract_model

    
class LaplaceModel(fem.abstract_model.AbstractModel):
    
    def element(self):
    
        return fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
    
    def weak_form_residual(self):

        dot, grad = fe.dot, fe.grad
        
        u = self.solution
        
        v = fe.TestFunction(self.solution.function_space())
        
        dx = fe.dx(degree = 2)
        
        return -(dot(grad(v), grad(u)))*dx
    