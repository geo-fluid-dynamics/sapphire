""" A steady incompressible Navier-Stokes-Boussinesq model class """
import firedrake as fe
import fempy.model

    
class Model(fempy.model.Model):
    
    def __init__(self, quadrature_degree, spatial_order):
        
        self.grashof_number = fe.Constant(1.)
        
        self.prandtl_number = fe.Constant(1.)
        
        super().__init__(
            quadrature_degree = quadrature_degree,
            spatial_order = spatial_order)
        
    def init_element(self):
    
        self.element = fe.MixedElement(
            fe.FiniteElement(
                "P", self.mesh.ufl_cell(), self.spatial_order - 1),
            fe.VectorElement(
                "P", self.mesh.ufl_cell(), self.spatial_order),
            fe.FiniteElement(
                "P", self.mesh.ufl_cell(), self.spatial_order - 1))
    
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
