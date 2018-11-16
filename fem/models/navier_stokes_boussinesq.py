""" A steady incompressible Navier-Stokes-Boussinesq model class. """
import firedrake as fe
import fem.model

    
class Model(fem.model.Model):
    
    def __init__(self):
    
        super().__init__()
        
        self.dynamic_viscosity = fe.Constant(1.)
        
        self.rayleigh_number = fe.Constant(1.)
        
        self.prandtl_number = fe.Constant(1.)
        
        ihat, jhat = self.unit_vectors()
        
        self.gravity_direction = fe.Constant(-jhat)
        
        self.pressure_penalty_factor = fe.Constant(1.e-7)
    
    def init_element(self):
    
        self.element = fe.MixedElement(
            fe.FiniteElement("P", self.mesh.ufl_cell(), 1),
            fe.VectorElement("P", self.mesh.ufl_cell(), 2),
            fe.FiniteElement("P", self.mesh.ufl_cell(), 1))
    
    def weak_form_residual(self):

        mu = self.dynamic_viscosity
        
        Ra = self.rayleigh_number
        
        Pr = self.prandtl_number
        
        ghat = self.gravity_direction
        
        gamma = self.pressure_penalty_factor
        
        inner, dot, grad, div, sym = \
            fe.inner, fe.dot, fe.grad, fe.div, fe.sym
        
        p, u, T = fe.split(self.solution)
        
        psi_p, psi_u, psi_T = fe.TestFunctions(self.solution.function_space())
        
        mass = psi_p*div(u)
        
        momentum = dot(psi_u, dot(grad(u), u) + Ra/Pr*T*ghat) \
            - div(psi_u)*p + 2.*mu*inner(sym(grad(psi_u)), sym(grad(u)))
        
        energy = psi_T*dot(u, grad(T)) + dot(grad(psi_T), 1./Pr*grad(T))
        
        stabilization = psi_p*gamma*p
        
        return mass + momentum + energy + stabilization
