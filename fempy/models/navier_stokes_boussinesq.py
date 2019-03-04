""" A steady incompressible Navier-Stokes-Boussinesq model class """
import firedrake as fe
import fempy.model

    
class Model(fempy.model.Model):
    
    def __init__(self, *args, mesh, element_degree, **kwargs):
        
        self.grashof_number = fe.Constant(1.)
        
        self.prandtl_number = fe.Constant(1.)
        
        scalar = fe.FiniteElement("P", mesh.ufl_cell(), element_degree)
        
        vector = fe.VectorElement("P", mesh.ufl_cell(), element_degree + 1)
        
        element = fe.MixedElement(scalar, vector, scalar)
        
        super().__init__(*args, mesh, element, **kwargs)
        
    def init_weak_form_residual(self):
        
        Gr = self.grashof_number
        
        Pr = self.prandtl_number
        
        ihat, jhat = self.unit_vectors()
        
        self.gravity_direction = fe.Constant(-jhat)
        
        ghat = self.gravity_direction
        
        inner, dot, grad, div, sym = \
            fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
        p, u, T = fe.split(self.solution)
        
        psi_p, psi_u, psi_T = fe.TestFunctions(self.solution.function_space())
        
        mass = psi_p*div(u)
        
        momentum = dot(psi_u, grad(u)*u + Gr*T*ghat) \
            - div(psi_u)*p + 2.*inner(sym(grad(psi_u)), sym(grad(u)))
        
        energy = psi_T*dot(u, grad(T)) + dot(grad(psi_T), 1./Pr*grad(T))
        
        self.weak_form_residual = mass + momentum + energy
        
    def strong_form_residual(self, solution):
            
            Gr = self.grashof_number
            
            Pr = self.prandtl_number
            
            ghat = self.gravity_direction
            
            grad, dot, div, sym = fe.grad, fe.dot, fe.div, fe.sym
            
            p, u, T = solution
            
            r_p = div(u)
            
            r_u = grad(u)*u + grad(p) - 2.*div(sym(grad(u))) + Gr*T*ghat
            
            r_T = dot(u, grad(T)) - 1./Pr*div(grad(T))
            
            return r_p, r_u, r_T
