""" **navier_stokes_model.py** 
implement a steady incompressible Navier-Stokes model class. 
"""
import firedrake as fe
import fem.abstract_model

    
class NavierStokesBoussinesqModel(fem.abstract_model.AbstractModel):
    
    def element(self):
    
        return fe.MixedElement(
            fe.FiniteElement("P", self.mesh.ufl_cell(), 1),
            fe.VectorElement("P", self.mesh.ufl_cell(), 2),
            fe.FiniteElement("P", self.mesh.ufl_cell(), 1))
    
    def weak_form_residual(self):

        mu = fe.Constant(self.residual_parameters["dynamic_viscosity"])
        
        Ra = fe.Constant(self.residual_parameters["rayleigh_number"])
        
        Pr = fe.Constant(self.residual_parameters["prandtl_number"])
        
        ghat = fe.Constant(self.residual_parameters["gravity_direction"])
        
        gamma = fe.Constant(self.residual_parameters["pressure_penalty_factor"])
        
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
