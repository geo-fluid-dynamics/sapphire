""" A convection-diffusion simulation class """
import firedrake as fe
import sapphire.simulation


class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args,
            advection_velocity,
            diffusion_coefficient = 1.,
            element_degree = 1,
            **kwargs):
        
        if "solution" not in kwargs:
            
            mesh = kwargs["mesh"]
            
            del kwargs["mesh"]
            
            element = fe.FiniteElement("P", mesh.ufl_cell(), element_degree)
            
            kwargs["solution"] = fe.Function(fe.FunctionSpace(mesh, element))
            
        self.diffusion_coefficient = fe.Constant(diffusion_coefficient)
        
        self.advection_velocity = advection_velocity(mesh)
    
        super().__init__(*args,
            **kwargs)
    
    def weak_form_residual(self):
        
        u = self.solution
        
        v = fe.TestFunction(self.solution_space)
        
        a = self.advection_velocity
        
        d = self.diffusion_coefficient
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        dot, grad, div = fe.dot, fe.grad, fe.div
        
        return (v*dot(a, grad(u)) + dot(grad(v), d*grad(u)))*dx
        
    def time_discrete_terms(self):
    
        return None
