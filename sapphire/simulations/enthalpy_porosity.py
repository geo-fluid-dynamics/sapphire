"""A simulation class using the enthalpy-porosity method.

Use this for convection-coupled melting and solidification.

Dirichlet BC's should not be placed on the pressure.
The returned pressure solution will always have zero mean.

Non-homogeneous Neumann BC's are not implemented.
"""
import firedrake as fe
import sapphire.simulation
import sapphire.simulations.enthalpy
import sapphire.simulations.unsteady_navier_stokes_boussinesq
import sapphire.continuation


dot, inner, grad, div, sym = fe.dot, fe.inner, fe.grad, fe.div, fe.sym

def phase_dependent_material_property(solid_to_liquid_ratio):
    
    a_sl = solid_to_liquid_ratio
    
    def a(phil):
    
        return a_sl + (1. - a_sl)*phil
    
    return a  


default_solver_parameters =  {
    "snes_monitor": None,
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxstep": 1.,
    "snes_linesearch_damping": 1.,
    "snes_atol": 1.e-9,
    "snes_stol": 1.e-9,
    "snes_rtol": 0.,
    "snes_max_it": 24,
    "ksp_type": "preonly", 
    "pc_type": "lu", 
    "pc_factor_mat_solver_type": "mumps",
    "mat_type": "aij"}

class Simulation(
        sapphire.simulations.unsteady_navier_stokes_boussinesq.Simulation):
    
    def __init__(
            self, 
            *args,
            stefan_number = 1.,
            liquidus_temperature = 0.,
            density_solid_to_liquid_ratio = 1.,
            heat_capacity_solid_to_liquid_ratio = 1.,
            thermal_conductivity_solid_to_liquid_ratio = 1.,
            solid_velocity_relaxation_factor = 1.e-12,
            liquidus_smoothing_factor = 0.01,
            solver_parameters = default_solver_parameters,
            **kwargs):
        
        self.stefan_number = fe.Constant(stefan_number)
        
        self.liquidus_temperature = fe.Constant(liquidus_temperature)
        
        self.density_solid_to_liquid_ratio = fe.Constant(
            density_solid_to_liquid_ratio)
        
        self.heat_capacity_solid_to_liquid_ratio = fe.Constant(
            heat_capacity_solid_to_liquid_ratio)
        
        self.thermal_conductivity_solid_to_liquid_ratio = fe.Constant(
            thermal_conductivity_solid_to_liquid_ratio)
        
        self.solid_velocity_relaxation_factor = fe.Constant(
            solid_velocity_relaxation_factor)
        
        self.liquidus_smoothing_factor = fe.Constant(
            liquidus_smoothing_factor)
        
        self.smoothing_sequence = None
        
        super().__init__(*args,
            solver_parameters = solver_parameters,
            **kwargs)
    
    def liquid_volume_fraction(self, temperature):
        
        return sapphire.simulations.enthalpy.Simulation.\
            liquid_volume_fraction(self, temperature = temperature)
    
    def solid_velocity_relaxation(self, temperature):
        
        T = temperature
        
        phil = self.liquid_volume_fraction(temperature = T)
        
        phis = 1. - phil
        
        tau = self.solid_velocity_relaxation_factor
        
        return 1./tau*phis
        
    def momentum(self):
        
        _, u, T = fe.split(self.solution)
        
        d = self.solid_velocity_relaxation(temperature = T)
        
        _, psi_u, _ = fe.TestFunctions(self.solution_space)
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return super().momentum() + dot(psi_u, d*u)*dx
    
    def energy(self):
        
        Pr = self.prandtl_number
        
        Ste = self.stefan_number
        
        _, u, T = fe.split(self.solution)
        
        phil = self.liquid_volume_fraction(temperature = T)
        
        rho_sl = self.density_solid_to_liquid_ratio
        
        c_sl = self.heat_capacity_solid_to_liquid_ratio
        
        C = phase_dependent_material_property(rho_sl*c_sl)(phil)
        
        k_sl = self.thermal_conductivity_solid_to_liquid_ratio
        
        k = phase_dependent_material_property(k_sl)(phil)
        
        CT_t, phil_t = self.extra_time_discrete_terms()
        
        _, _, psi_T = fe.TestFunctions(self.solution_space)
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        return (psi_T*(CT_t + 1./Ste*phil_t + dot(u, grad(C*T))) \
            + 1./Pr*dot(grad(psi_T), k*grad(T)))*dx
        
    def extra_time_discrete_terms(self):
        
        temperature_solutions = []
        
        for solution in self.solutions:
        
            temperature_solutions.append(fe.split(solution)[2])
        
        def phil(T):
        
            return self.liquid_volume_fraction(temperature = T)
            
        rho_sl = self.density_solid_to_liquid_ratio
        
        c_sl = self.heat_capacity_solid_to_liquid_ratio
        
        C = phase_dependent_material_property(rho_sl*c_sl)
        
        CT_t = sapphire.time_discretization.bdf(
            [C(phil(T))*T for T in temperature_solutions],
            timestep_size = self.timestep_size)
        
        phil_t = sapphire.time_discretization.bdf(
            [phil(T) for T in temperature_solutions],
            timestep_size = self.timestep_size)
        
        return CT_t, phil_t
        
    def solve_with_over_regularization(self):
        
        return sapphire.continuation.solve_with_over_regularization(
            solve = self.solve,
            solution = self.solution,
            regularization_parameter = self.liquidus_smoothing_factor)
            
    def solve_with_bounded_regularization_sequence(self):
        
        return sapphire.continuation.\
            solve_with_bounded_regularization_sequence(
                solve = self.solve,
                solution = self.solution,
                backup_solution = self.backup_solution,
                regularization_parameter = self.liquidus_smoothing_factor,
                initial_regularization_sequence = self.smoothing_sequence)
                
    def solve_with_auto_smoothing(self):
        
        sigma = self.liquidus_smoothing_factor.__float__()
        
        if self.smoothing_sequence is None:
        
            given_smoothing_sequence = False
            
        else:
        
            given_smoothing_sequence = True
        
        
        if not given_smoothing_sequence:
            # Find an over-regularization that works.
            self.solution, sigma_max = self.solve_with_over_regularization()
            
            if sigma_max == sigma:
                # No over-regularization was necessary.
                return self.solution
                
            else:
                # A working over-regularization was found, which becomes
                # the upper bound of the sequence.
                self.smoothing_sequence = (sigma_max, sigma)
                
                
        # At this point, either a smoothing sequence has been provided,
        # or a working upper bound has been found.
        # Next, a viable sequence will be sought.
        try:
            
            self.solution, self.smoothing_sequence = \
                self.solve_with_bounded_regularization_sequence()
                
        except fe.exceptions.ConvergenceError as error: 
            
            if given_smoothing_sequence:
                # Try one more time without using the given sequence.
                # This is sometimes useful after solving some time steps
                # with a previously successful regularization sequence
                # that is not working for a new time step.
                self.solution = self.solution.assign(self.solutions[1])
                
                self.solution, smax = self.solve_with_over_regularization()
                
                self.smoothing_sequence = (smax, sigma)
                
                self.solution, self.smoothing_sequence = \
                    self.solve_with_bounded_regularization_sequence()
                    
            else:
                
                raise error
                
                
        # For debugging purposes, ensure that the problem was solved with the 
        # correct regularization and that the simulation's attribute for this
        # has been set to the correct value before returning.
        assert(self.liquidus_smoothing_factor.__float__() == sigma)
        
        return self.solution
        
    def kwargs_for_writeplots(self):
        
        p, u, T = self.solution.split()
        
        phil = fe.interpolate(
            self.liquid_volume_fraction(temperature = T),
            T.function_space())
        
        return {
            "fields": (p, u, T, phil),
            "labels": ("p", "\\mathbf{u}", "T", "\\phi_l"),
            "names": ("p", "u", "T", "phil"),
            "plotfuns": (fe.tripcolor, fe.quiver, fe.tripcolor, fe.tripcolor)}
            
    def run(self, *args, **kwargs):
        
        return super().run(*args,
            solve = self.solve_with_auto_smoothing,
            **kwargs)
           
    def postprocess(self):
        
        p, u, T = self.solution.split()
        
        dx = fe.dx(degree = self.quadrature_degree)
        
        div = fe.div
        
        self.velocity_divergence = fe.assemble(div(u)*dx)
        
        phil = self.liquid_volume_fraction(temperature = T)
        
        self.liquid_area = fe.assemble(phil*dx)
        
        return self


diff = fe.diff

def strong_residual(sim, solution):
    
    r_p, r_u, r_T = sapphire.simulations.unsteady_navier_stokes_boussinesq.\
        strong_residual(sim = sim, solution = solution)
    
    _, u, T = solution
    
    t = sim.time
    
    
    d = sim.solid_velocity_relaxation(temperature = T)
    
    r_u += d*u
    
    
    Pr = sim.prandtl_number
    
    Ste = sim.stefan_number
    
    phil = sim.liquid_volume_fraction(temperature = T)
    
    rho_sl = sim.density_solid_to_liquid_ratio
    
    c_sl = sim.heat_capacity_solid_to_liquid_ratio
    
    C = phase_dependent_material_property(rho_sl*c_sl)(phil)
    
    k_sl = sim.thermal_conductivity_solid_to_liquid_ratio
    
    k = phase_dependent_material_property(k_sl)(phil)
    
    r_T = diff(C*T, t) + 1./Ste*diff(phil, t) + dot(u, grad(C*T)) \
        - 1./Pr*div(k*grad(T))
    
    return r_p, r_u, r_T
