""" An unsteady incompressible Navier-Stokes model class """
import firedrake as fe
import fempy.unsteady_model

    
class Model(fempy.unsteady_model.UnsteadyModel):
    
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
        
        u_t, _ = self.time_discrete_terms
        
        psi_u, psi_p = fe.TestFunctions(self.solution.function_space())
        
        mass = psi_p*div(u)
        
        momentum = dot(psi_u, u_t + grad(u)*u) - div(psi_u)*p + \
            2.*inner(sym(grad(psi_u)), sym(grad(u)))
        
        self.weak_form_residual = mass + momentum
    
    def strong_form_residual(self, solution):
    
        diff, grad, div, sym = fe.diff, fe.grad, fe.div, fe.sym
        
        u, p = solution
        
        t = self.time
        
        r_u = diff(u, t) + grad(u)*u + grad(p) - 2.*div(sym(grad(u)))
        
        r_p = div(u)
        
        return r_u, r_p
        