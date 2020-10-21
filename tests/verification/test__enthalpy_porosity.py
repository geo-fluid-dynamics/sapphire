"""Verify accuracy of the enthalpy-porosity solver

for convection-coupled phase-change.
"""
import sys
import pathlib
import firedrake as fe 
import sapphire.mms
import tests.verification.test__unsteady_navier_stokes_boussinesq
from sapphire.simulations.enthalpy_porosity import Simulation


def strong_residual(sim, solution):
    
    r_p, r_u, _ = tests.verification.test__unsteady_navier_stokes_boussinesq.\
        strong_residual(sim = sim, solution = solution)
    
    _, u, T = solution
    
    t = sim.time
    
    
    d = sim.solid_velocity_relaxation(temperature = T)
    
    r_u += d*u
    
    
    Re = sim.reynolds_number
    
    Pr = sim.prandtl_number
    
    Ste = sim.stefan_number
    
    phi_l = sim.liquid_volume_fraction(temperature = T)
    
    rho_sl = sim.density_solid_to_liquid_ratio
    
    c_sl = sim.heat_capacity_solid_to_liquid_ratio
    
    C = sim.volumetric_heat_capacity(temperature = T)
    
    k = sim.thermal_conductivity(temperature = T)
    
    diff, dot, grad, div, sym = \
        fe.diff, fe.dot, fe.grad, fe.div, fe.sym
    
    r_T = diff(C*T, t) + 1/Ste*diff(phi_l, t) + dot(u, grad(C*T)) \
        - 1/(Re*Pr)*div(k*grad(T))
    
    return r_p, r_u, r_T


pi, sin, exp = fe.pi, fe.sin, fe.exp

def space_verification_solution(sim):
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    ihat, jhat = sim.unit_vectors
    
    u = exp(0.5)*(sin(2*pi*x)*sin(pi*y)*ihat + 
                  sin(pi*x)*sin(2*pi*y)*jhat)
    
    p = -sin(pi*x - pi/2)*sin(2*pi*y - pi/2)
    
    T = 0.5*sin(2*pi*x)*sin(pi*y)*(1 - exp(-0.5))
    
    mean_pressure = fe.assemble(p*fe.dx)
    
    p -= mean_pressure
    
    return p, u, T
    
    
def time_verification_solution(sim):
    
    t = sim.time
    
    x, y = fe.SpatialCoordinate(sim.mesh)
    
    ihat, jhat = sim.unit_vectors
    
    u = exp(t/2)*(sin(2*pi*x)*sin(pi*y)*ihat +
                  sin(pi*x)*sin(2*pi*y)*jhat)
    
    p = -sin(pi*x - pi/2)*sin(2*pi*y - pi/2)
    
    T = 0.5*sin(2*pi*x)*sin(pi*y)*(1 - exp(-0.5*t**2))
    
    mean_pressure = fe.assemble(p*fe.dx)
    
    p -= mean_pressure
    
    return p, u, T
    
    
class UnitSquareUniformMeshSimulation(Simulation):
    
    def __init__(self, *args,
            meshcell_size,
            **kwargs):
        
        n = int(round(1/meshcell_size))
        
        kwargs['mesh'] = fe.UnitSquareMesh(n, n)
        
        super().__init__(*args, **kwargs)
    
    
def dirichlet_boundary_conditions(sim, manufactured_solution):
    """Apply velocity and temperature Dirichlet BC's on every boundary.
    
    Do not apply Dirichlet BC's on the pressure.
    """
    _, u, T = manufactured_solution
    
    return [
        fe.DirichletBC(sim.solution_subspaces['u'], u, 'on_boundary'),
        fe.DirichletBC(sim.solution_subspaces['T'], T, 'on_boundary')]
    

sim_kwargs = {
    'reynolds_number': 20,
    'rayleigh_number': 2.5e6,
    'prandtl_number': 7.0,
    'stefan_number': 0.13,
    'density_solid_to_liquid_ratio': 0.92,
    'heat_capacity_solid_to_liquid_ratio': 0.50,
    'thermal_conductivity_solid_to_liquid_ratio': 3.8,
    'liquidus_smoothing_factor': 0.1,
    'solid_velocity_relaxation_factor': 1.e-12,
    'quadrature_degree': 4}


endtime = 1

def test__verify_second_order_spatial_convergence_via_mms(tmpdir):
    
    sim_kwargs['taylor_hood_pressure_degree'] = 1
    
    sim_kwargs['temperature_degree'] = 2
    
    sim_kwargs['timestep_size'] = endtime
    
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = 'meshcell_size',
        discretization_parameter_values = [1/n for n in (4, 8, 16)],
        Simulation = UnitSquareUniformMeshSimulation,
        sim_kwargs = sim_kwargs,
        strong_residual = strong_residual,
        manufactured_solution = space_verification_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        norms = ('L2', 'H1', 'H1'),
        expected_orders = (None, 2, 2),
        decimal_places = 1,
        endtime = endtime)


def test__verify_second_order_temporal_convergence_via_mms(tmpdir):
    
    sim_kwargs['taylor_hood_pressure_degree'] = 2
    
    sim_kwargs['temperature_degree'] = 3
    
    sim_kwargs['mesh'] = fe.UnitSquareMesh(16, 16)
    
    sim_kwargs['time_stencil_size'] = 3
    
    sapphire.mms.verify_order_of_accuracy(
        discretization_parameter_name = 'timestep_size',
        discretization_parameter_values = (1/2, 1/4, 1/8, 1/16, 1/32),
        Simulation = UnitSquareUniformMeshSimulation,
        sim_kwargs = sim_kwargs,
        strong_residual = strong_residual,
        manufactured_solution = time_verification_solution,
        dirichlet_boundary_conditions = dirichlet_boundary_conditions,
        norms = (None, 'L2', 'L2'),
        expected_orders = (None, 2, 2),
        decimal_places = 1,
        endtime = endtime)
