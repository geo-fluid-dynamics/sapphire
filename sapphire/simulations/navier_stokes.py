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
            taylor_hood_pressure_degree = 1,
            **kwargs):
        
        if "solution" not in kwargs:
            
            mesh = kwargs["mesh"]
            
            del kwargs["mesh"]
            
            d = taylor_hood_pressure_degree
            
            element = fe.MixedElement(
                fe.FiniteElement("P", mesh.ufl_cell(), d),
                fe.VectorElement("P", mesh.ufl_cell(), d + 1))
                
            kwargs["solution"] = fe.Function(fe.FunctionSpace(mesh, element))
        
        self.reynolds_number = fe.Constant(reynolds_number)
        
        super().__init__(*args, fieldnames=("p", "u"), **kwargs)
    
    def mass(self):
        
        u = self.solution_fields["u"]
        
        psi_p = self.test_functions["p"]
        
        return psi_p*div(u)*self.dx
    
    def momentum(self):
        
        p, u = self.solution_fields["p"], self.solution_fields["u"]
        
        Re = self.reynolds_number
        
        psi_u = self.test_functions["u"]
        
        return (dot(psi_u, grad(u)*u) - div(psi_u)*p + \
            2./Re*inner(sym(grad(psi_u)), sym(grad(u))))*self.dx
    
    def weak_form_residual(self):
        
        return self.mass() + self.momentum()
    
    def solve(self):
        
        self.solution = super().solve()
        
        print("Subtracting mean pressure")
        
        p = self.solution_fields["p"]
        
        mean_pressure = fe.assemble(p*self.dx)
        
        p = self.solution_subfunctions["p"]
        
        p = p.assign(p - mean_pressure)
        
        print("Done subtracting mean pressure")
        
        return self.solution
    
    def nullspace(self):
        """Inform solver that pressure solution is not unique.
        
        It is only defined up to adding an arbitrary constant.
        """
        return fe.MixedVectorSpaceBasis(
            self.solution_space,
            (fe.VectorSpaceBasis(constant=True),
             self.solution_subspaces["u"]))
        