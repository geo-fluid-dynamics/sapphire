""" An enthalpy model class for melting and solidification of binary alloys"""
import firedrake as fe
import fempy.unsteady_model
import fempy.autosmooth
import matplotlib.pyplot as plt


class Model(fempy.unsteady_model.Model):
    
    def __init__(self):
        
        self.stefan_number = fe.Constant(1.)
        
        self.lewis_number = fe.Constant(1.)
        
        self.pure_liquidus_temperature = fe.Constant(0.)
        
        self.liquidus_slope = fe.Constant(-0.1)
        
        self.solid_concentration = fe.Constant(0.)
        
        self.smoothing = fe.Constant(1./64.)
        
        super().__init__()
        
        self.backup_solution = fe.Function(self.solution)
        
        self.smoothing_sequence = None
        
    def init_element(self):
    
        P1 = fe.FiniteElement("P", self.mesh.ufl_cell(), 1)
        
        self.element = fe.MixedElement(P1, P1)
        
    def liquidus_temperature(self, Cl):
        
        T_m = self.pure_liquidus_temperature
        
        m_L = self.liquidus_slope
        
        return T_m + m_L*Cl
        
    def porosity(self, T, Cl):
        
        T_L = self.liquidus_temperature(Cl)
        
        s = self.smoothing
        
        tanh = fe.tanh
        
        return 0.5*(1. + tanh((T - T_L)/s))
    
    def init_time_discrete_terms(self):
        """ Implicit Euler finite difference scheme """
        T, Cl = self.solution
        
        T_n, Cl_n = self.initial_values
        
        Delta_t = self.timestep_size
        
        T_t = (T - T_n)/Delta_t
        
        Cl_t = (Cl - Cl_n)/Delta_t
        
        phil = self.porosity
        
        phil_t = (phil(T, Cl) - phil(T_n, Cl_n))/Delta_t
        
        self.time_discrete_terms = T_t, Cl_t, phil_t
    
    def init_weak_form_residual(self):
        
        T, Cl = self.solution
        
        Ste = self.stefan_number
        
        Le = self.lewis_number
        
        Cs = self.solid_concentration
        
        phil = self.porosity(T, Cl)
        
        T_t, Cl_t, phil_t = self.time_discrete_terms
        
        psi_T, psi_Cl = fe.TestFunctions(self.function_space)
        
        dot, grad = fe.dot, fe.grad
        
        enthalpy =  psi_T*(T_t + 1./Ste*phil_t) + dot(grad(psi_T), grad(T))
        
        concentration = psi_Cl*(phil*Cl_t + (Cl - Cs)*phil_t) \
            + 1./Le*dot(grad(psi_Cl), phil*grad(Cl))
            
        self.weak_form_residual = enthalpy + concentration
        
    def solve(self):
        
        fempy.autosmooth.solve(self)
        
    def plot(self):
    
        V = fe.FunctionSpace(
            self.mesh, fe.FiniteElement("P", self.mesh.ufl_cell(), 1))
        
        T, Cl = self.solution.split()
        
        _phil = self.porosity(T, Cl)
        
        phil = fe.interpolate(_phil, V)
        
        C = fe.interpolate(_phil*Cl, V)
        
        timestr = str(self.time.__float__())
        
        self.output_directory_path.mkdir(
                parents = True, exist_ok = True)
        
        for f, label, filename in zip(
                (T, Cl, C, phil),
                ("T", "C_l", "C", "\\phi_l"),
                ("T", "Cl", "C", "phil")):
            
            fe.plot(f)
            
            plt.xlabel(r"$x$")
            
            plt.title(r"$" + label + 
                ", t = " + timestr + "$")
            
            filepath = self.output_directory_path.joinpath(filename + 
                "_t" + timestr.replace(".", "p")).with_suffix(".png")
            
            print("Writing plot to " + str(filepath))
            
            plt.savefig(str(filepath))
            
            plt.close()
            