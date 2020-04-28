"""Provides a simulation class governed by Navier-Stokes-Boussinesq.

This can be used to simulate natural convection,
e.g the heat-driven cavity.

Dirichlet BC's should not be placed on the pressure.
The returned pressure solution will always have zero mean.

Non-homogeneous Neumann BC's are not implemented for the velocity.
"""
import firedrake as fe
import sapphire.simulation


inner, dot, grad, div, sym = \
    fe.inner, fe.dot, fe.grad, fe.div, fe.sym

class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args,
            element_degrees = (1, 2, 2),
            grashof_number = 1.,
            prandtl_number = 1.,
            **kwargs):
        
        if "solution" not in kwargs:
            
            mesh = kwargs["mesh"]
            
            del kwargs["mesh"]
            
            element = fe.MixedElement(
                fe.FiniteElement("P", mesh.ufl_cell(), element_degrees[0]),
                fe.VectorElement("P", mesh.ufl_cell(), element_degrees[1]),
                fe.FiniteElement("P", mesh.ufl_cell(), element_degrees[2]))
            
            kwargs["solution"] = fe.Function(fe.FunctionSpace(mesh, element))
            
        self.grashof_number = fe.Constant(grashof_number)
        
        self.prandtl_number = fe.Constant(prandtl_number)
        
        super().__init__(*args, **kwargs)
    
    def mass(self):

        _, u, _ = fe.split(self.solution)
        
        psi_p, _, _ = fe.TestFunctions(self.solution_space)
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return psi_p*div(u)*dx
        
    def momentum(self):
        
        p, u, T = fe.split(self.solution)
        
        _, psi_u, _ = fe.TestFunctions(self.solution_space)
        
        b = self.buoyancy(temperature = T)
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return (dot(psi_u, grad(u)*u + b) - div(psi_u)*p + \
            2.*inner(sym(grad(psi_u)), sym(grad(u))))*dx
        
    def energy(self):
    
        Pr = self.prandtl_number
        
        _, u, T = fe.split(self.solution)
        
        _, _, psi_T = fe.TestFunctions(self.solution_space)
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return (psi_T*dot(u, grad(T)) + dot(grad(psi_T), 1./Pr*grad(T)))*dx
    
    def weak_form_residual(self):
        
        return self.mass() + self.momentum() + self.energy()
    
    def solve(self):
        
        self.solution = super().solve()
        
        print("Subtracting mean pressure")
        
        p, _, _ = self.solution.split()
        
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
            W, [fe.VectorSpaceBasis(constant=True), W.sub(1), W.sub(2)])
        
    def buoyancy(self, temperature):
        """Linear Boussinesq buoyancy"""
        T = temperature
        
        Gr = self.grashof_number
        
        ghat = fe.Constant(-self.unit_vectors()[1])
        
        return Gr*T*ghat
        
    def time_discrete_terms(self):
    
        return None


def strong_residual(sim, solution):
    
    Pr = sim.prandtl_number
    
    p, u, T = solution
    
    b = sim.buoyancy(temperature = T)
    
    r_p = div(u)
    
    r_u = grad(u)*u + grad(p) - 2.*div(sym(grad(u))) + b
    
    r_T = dot(u, grad(T)) - 1./Pr*div(grad(T))
    
    return r_p, r_u, r_T
    