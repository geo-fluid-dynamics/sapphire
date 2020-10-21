""" Water freezing benchmark simulation

based on

    @article{danaila2014newton,
        title={A {N}ewton method with adaptive finite elements 
            for solving phase-change problems with natural convection},
        author={Danaila, Ionut and Moglan, Raluca and Hecht, 
            Fr{\'e}d{\'e}ric and Le Masson, St{\'e}phane},
        journal={Journal of Computational Physics},
        volume={274},
        pages={826--840},
        year={2014},
        publisher={Elsevier}
    }
"""
import firedrake as fe
import sapphire.simulations.examples.heat_driven_cavity_with_water
import sapphire.simulations.enthalpy_porosity


class Simulation(sapphire.simulations.enthalpy_porosity.Simulation):

    def __init__(self, *args,
            mesh_dimensions = (24, 24),
            taylor_hood_pressure_degree = 1,
            temperature_degree = 2,
            reference_temperature_range__degC = 10.,
            hotwall_temperature = 1.,
            coldwall_temperature_before_freezing = 0.,
            coldwall_temperature_during_freezing = -1.,
            reynolds_number = 1.,
            rayleigh_number = 2.518084e6, 
            prandtl_number = 6.99,
            stefan_number = 0.125,
            liquidus_temperature = 0.,
            density_solid_to_liquid_ratio = 916.70/999.84,
            heat_capacity_solid_to_liquid_ratio = 0.500,
            thermal_conductivity_solid_to_liquid_ratio = 2.14/0.561,
            output_directory_path = "output/",
            **kwargs):
        
        iv_sim = sapphire.simulations.examples.\
            heat_driven_cavity_with_water.Simulation(
                mesh_dimensions = mesh_dimensions,
                taylor_hood_pressure_degree = taylor_hood_pressure_degree,
                temperature_degree = temperature_degree,
                reference_temperature_range__degC = \
                    reference_temperature_range__degC,
                hotwall_temperature = hotwall_temperature,
                coldwall_temperature = coldwall_temperature_before_freezing,
                reynolds_number = reynolds_number,
                rayleigh_number = rayleigh_number,
                prandtl_number = prandtl_number,
                output_directory_path = output_directory_path)
        
        iv_sim.solution = iv_sim.solve_with_continuation_on_grashof_number()
        
        self.reference_temperature_range__degC = fe.Constant(
            reference_temperature_range__degC)
        
        self.hotwall_temperature = fe.Constant(
            hotwall_temperature)
        
        self.coldwall_temperature = fe.Constant(
            coldwall_temperature_during_freezing)\
            
        self.hotwall_id = iv_sim.hotwall_id
        
        self.coldwall_id = iv_sim.coldwall_id
        
        super().__init__(
            *args,
            solution = iv_sim.solution,
            reynolds_number = reynolds_number,
            rayleigh_number = rayleigh_number,
            prandtl_number = prandtl_number,
            stefan_number = stefan_number,
            liquidus_temperature = liquidus_temperature,
            density_solid_to_liquid_ratio = density_solid_to_liquid_ratio,
            heat_capacity_solid_to_liquid_ratio = \
                heat_capacity_solid_to_liquid_ratio,
            thermal_conductivity_solid_to_liquid_ratio = \
                thermal_conductivity_solid_to_liquid_ratio,
            output_directory_path = output_directory_path,
            **kwargs)
    
    def buoyancy(self, temperature):
    
        return sapphire.simulations.examples.heat_driven_cavity_with_water.\
            Simulation.buoyancy(self, temperature = temperature)
    
    def dirichlet_boundary_conditions(self):
    
        return sapphire.simulations.examples.heat_driven_cavity_with_water.\
            Simulation.dirichlet_boundary_conditions(self)
            