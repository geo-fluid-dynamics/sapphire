"""Provides a simulation class governed by Navier-Stokes-Boussinesq.

This can be used to simulate natural convection,
e.g the heat-driven cavity.

Dirichlet BC's should not be placed on the pressure.
The returned pressure solution will always have zero mean.

Non-homogeneous Neumann BC's are not implemented for the velocity.
"""
import firedrake as fe
import sapphire.simulation


def element(cell, taylor_hood_pressure_degree, temperature_degree):
    
    return fe.MixedElement(
        fe.FiniteElement("P", cell, taylor_hood_pressure_degree),
        fe.VectorElement("P", cell, taylor_hood_pressure_degree + 1),
        fe.FiniteElement("P", cell, temperature_degree))

inner, dot, grad, div, sym = \
    fe.inner, fe.dot, fe.grad, fe.div, fe.sym

class Simulation(sapphire.simulation.Simulation):
    
    def __init__(self, *args,
            taylor_hood_pressure_degree = 1,
            temperature_degree = 2,
            reynolds_number = 1.,
            rayleigh_number = 1.,
            prandtl_number = 1.,
            **kwargs):
        
        if "solution" not in kwargs:
            
            mesh = kwargs["mesh"]
            
            del kwargs["mesh"]
            
            kwargs["solution"] = fe.Function(fe.FunctionSpace(
                mesh,
                element(
                    mesh.ufl_cell(),
                    taylor_hood_pressure_degree,
                    temperature_degree)))
            
        self.reynolds_number = fe.Constant(reynolds_number)
        
        self.rayleigh_number = fe.Constant(rayleigh_number)
        
        self.prandtl_number = fe.Constant(prandtl_number)
        
        super().__init__(*args, fieldnames=("p", "u", "T"), **kwargs)
    
    def mass(self):

        u = self.solution_fields["u"]
        
        psi_p = self.test_functions["p"]
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return psi_p*div(u)*dx
        
    def momentum(self):
        
        p, u, T = fe.split(self.solution)
        
        psi_u = self.test_functions["u"]
        
        b = self.buoyancy(temperature = T)
        
        Re = self.reynolds_number
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return (dot(psi_u, grad(u)*u + b) - div(psi_u)*p + \
            2./Re*inner(sym(grad(psi_u)), sym(grad(u))))*dx
        
    def energy(self):
    
        Re = self.reynolds_number
        
        Pr = self.prandtl_number
        
        u = self.solution_fields["u"]
        
        T = self.solution_fields["T"]
        
        psi_T = self.test_functions["T"]
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return (psi_T*dot(u, grad(T)) + dot(grad(psi_T), 1./(Re*Pr)*grad(T)))*dx
    
    def weak_form_residual(self):
        
        return self.mass() + self.momentum() + self.energy()
    
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
        W = self.solution_space
        
        return fe.MixedVectorSpaceBasis(
            W, 
            [fe.VectorSpaceBasis(constant=True), 
             self.solution_subspaces["u"],
             self.solution_subspaces["T"]])
        
    def buoyancy(self, temperature):
        """Linear Boussinesq buoyancy"""
        T = temperature
        
        Re = self.reynolds_number
        
        Ra = self.rayleigh_number
        
        Pr = self.prandtl_number
        
        ghat = fe.Constant(-self.unit_vectors[1])
        
        return Ra/(Pr*Re**2)*T*ghat
        