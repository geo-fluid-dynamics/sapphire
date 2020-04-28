"""Provides a simulation class governed by steady Navier-Stokes. 

This can be used to simulate incompressible flow,
e.g. the lid-driven cavity.

Dirichlet BC's should not be placed on the pressure.
The returned pressure solution will always have zero mean.

Non-homogeneous Neumann BC's are not implemented.
"""
import firedrake as fe
import sapphire.simulation


inner, dot, grad, div, sym = \
    fe.inner, fe.dot, fe.grad, fe.div, fe.sym

    
class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args,
            reynolds_number,
            element_degrees = (2, 1),
            **kwargs):
        
        if "solution" not in kwargs:
            
            mesh = kwargs["mesh"]
            
            del kwargs["mesh"]
            
            element = fe.MixedElement(
                fe.VectorElement("P", mesh.ufl_cell(), element_degrees[0]),
                fe.FiniteElement("P", mesh.ufl_cell(), element_degrees[1]))
                
            kwargs["solution"] = fe.Function(fe.FunctionSpace(mesh, element))
        
        self.reynolds_number = fe.Constant(reynolds_number)
        
        super().__init__(*args, **kwargs)
    
    def mass(self):
        
        u, _ = fe.split(self.solution)
        
        _, psi_p = fe.TestFunctions(self.solution_space)
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return psi_p*div(u)*dx
    
    def momentum(self):
        
        u, p = fe.split(self.solution)
        
        Re = self.reynolds_number
        
        psi_u, _ = fe.TestFunctions(self.solution_space)
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return (dot(psi_u, grad(u)*u) - div(psi_u)*p + \
            2./Re*inner(sym(grad(psi_u)), sym(grad(u))))*dx
    
    def weak_form_residual(self):
        
        return self.mass() + self.momentum()
    
    def solve(self):
        
        self.solution = super().solve()
        
        print("Subtracting mean pressure")
        
        u, p = self.solution.split()
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        mean_pressure = fe.assemble(p*dx)
        
        p = p.assign(p - mean_pressure)
        
        print("Done subtracting mean pressure")
        
        return self.solution
    
    def nullspace(self):
        """Inform solver that pressure solution is not unique.
        
        It is only defined up to adding an arbitrary constant.
        """
        W = self.solution_space
        
        return fe.MixedVectorSpaceBasis(
            W, [W.sub(0), fe.VectorSpaceBasis(constant=True)])            
    
    def time_discrete_terms(self):
    
        return None


def strong_residual(sim, solution):
    
    u, p = solution
    
    Re = sim.reynolds_number
    
    r_u = grad(u)*u + grad(p) - 2./Re*div(sym(grad(u)))
    
    r_p = div(u)
    
    return r_u, r_p
    