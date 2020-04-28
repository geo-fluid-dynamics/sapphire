import firedrake as fe
import sapphire.simulations.examples.heat_driven_cavity


class Simulation(sapphire.simulations.examples.heat_driven_cavity.Simulation):
    
    def __init__(self, *args,
            reference_temperature_range__degC = 10.,
            grashof_number = 2.518084e6/6.99,
            prandtl_number = 6.99,
            hotwall_temperature = 1.,
            coldwall_temperature = 0.,
            solver_parameters = {
                "snes_monitor": None,
                "snes_type": "newtonls",
                "snes_max_it": 10,
                "ksp_type": "preonly", 
                "pc_type": "lu", 
                "mat_type": "aij",
                "pc_factor_mat_solver_type": "mumps"},
            **kwargs):
        
        self.reference_temperature_range__degC = fe.Constant(
            reference_temperature_range__degC)
            
        super().__init__(*args,
            grashof_number = grashof_number,
            prandtl_number = prandtl_number,
            solver_parameters = solver_parameters,
            hotwall_temperature = hotwall_temperature,
            coldwall_temperature = coldwall_temperature,
            **kwargs)
    
    def solve_with_continuation_on_grashof_number(self, *args, **kwargs):
    
        self.solution, _ = \
            sapphire.continuation.solve_with_bounded_regularization_sequence(
                solve = super().solve,
                solution = self.solution,
                backup_solution = fe.Function(self.solution),
                regularization_parameter = self.grashof_number,
                initial_regularization_sequence = (
                    0., self.grashof_number.__float__()))
        
        return self.solution
    
    def buoyancy(self, temperature):
        """ Eq. (25) from @cite{danaila2014newton} """
        T = temperature
        
        T_anomaly_degC = fe.Constant(4.0293)
        
        rho_anomaly_SI = fe.Constant(999.972)
        
        w_degC = fe.Constant(9.2793e-6)
        
        q = fe.Constant(1.894816)
        
        M = self.reference_temperature_range__degC
        
        def T_degC(T):
            """ T = T_degC/M """
            return M*T
        
        def rho_of_T_degC(T_degC):
            """ Eq. (24) from @cite{danaila2014newton} """
            return rho_anomaly_SI*(1. - w_degC*abs(T_degC - T_anomaly_degC)**q)
            
        def rho(T):
            
            return rho_of_T_degC(T_degC(T))
        
        beta = fe.Constant(6.91e-5)  # [K^-1]
        
        Gr = self.grashof_number
        
        ghat = fe.Constant(-self.unit_vectors()[1])
        
        rho_0 = rho(T = 0.)
        
        return Gr/(beta*M)*(rho_0 - rho(T))/rho_0*ghat
        