""" Example heat-driven cavity simulation governed by the Navier-Stokes-Boussinesq equations

The result is compared to data published in

    @article{wang2010comprehensive,
        author = {Ghia, Urmila and Ghia, K.N and Shin, C.T},
        year = {1982},
        month = {12},
        pages = {387-411},
        title = {High-Re solutions for incompressible flow using the
        Navier-Stokes equations and a multigrid method},
        volume = {48},
        journal = {Journal of Computational Physics},
        doi = {10.1016/0021-9991(82)90058-4}
    }
"""
from sapphire import Solution, Simulation, plot
from sapphire.examples.heat_driven_cavity import mesh, element, COMPONENT_NAMES, dirichlet_boundary_conditions, nullspace, solve, run
from sapphire.examples.heat_driven_cavity import residual as heat_driven_cavity_residual
from firedrake import Constant, dot, grad, FacetNormal, assemble, ds


def buoyancy_with_density_anomaly_of_water(solution: Solution):
    """Eq. (25) from @cite{danaila2014newton}"""
    T = solution.ufl_fields.T

    Re = solution.ufl_constants.reynolds_number

    Ra = solution.ufl_constants.rayleigh_number

    Pr = solution.ufl_constants.prandtl_number

    DeltaT = solution.ufl_constants.reference_temperature_range__degC

    ghat = Constant(-solution.unit_vectors[1])

    T_anomaly_degC = Constant(4.0293)

    rho_anomaly_SI = Constant(999.972)

    w_degC = Constant(9.2793e-6)

    q = Constant(1.894816)

    def T_degC(T):
        """ T = T_degC/DeltaT """
        return DeltaT*T

    def rho_of_T_degC(T_degC):
        """ Eq. (24) from @cite{danaila2014newton} """
        return rho_anomaly_SI*(1. - w_degC*abs(T_degC - T_anomaly_degC)**q)

    def rho(T):

        return rho_of_T_degC(T_degC(T))

    beta = Constant(6.91e-5)  # [K^-1]

    rho_0 = rho(T=0.)

    return Ra/(Pr*Re**2*beta*DeltaT)*(rho_0 - rho(T))/rho_0*ghat


def residual(solution: Solution):

    return heat_driven_cavity_residual(solution, buoyancy=buoyancy_with_density_anomaly_of_water)


def output(solution: Solution):

    plot(solution=solution, outdir_path="sapphire_output/heat_driven_cavity_with_water/plots/")


def run_simulation(
        reynolds_number=1.,
        rayleigh_number=2.518084e6,
        prandtl_number=6.99,
        hotwall_temperature=1.,
        coldwall_temperature=0.,
        reference_temperature_range__degC=10.,
        mesh_dimensions=(20, 20),
        taylor_hood_velocity_element_degree=2,
        temperature_element_degree=2
        ) -> Simulation:

    _mesh = mesh(mesh_dimensions)

    sim = Simulation(
        mesh=_mesh,
        element=element(_mesh.cell, taylor_hood_velocity_element_degree, temperature_element_degree),
        solution_component_names=COMPONENT_NAMES,
        ufl_constants={
            'reynolds_number': reynolds_number,
            'rayleigh_number': rayleigh_number,
            'prandtl_number': prandtl_number,
            'hotwall_temperature': hotwall_temperature,
            'coldwall_temperature': coldwall_temperature,
            'reference_temperature_range__degC': reference_temperature_range__degC},
        residual=residual,
        dirichlet_boundary_conditions=dirichlet_boundary_conditions,
        nullspace=nullspace,
        initial_times=None)

    return run(sim=sim, solve=solve, output=output)


def verify_default_simulation():

    sim = run_simulation()

    solution = sim.solutions[0]

    nhat = FacetNormal(solution.mesh.geometry)

    T = solution.ufl_fields.T

    coldwall_heatflux = assemble(dot(grad(T), nhat)*ds(subdomain_id=solution.mesh.boundaries['right']))

    expected_coldwall_heatflux = -8

    print("Integrated cold wall heat flux = {} (Expected {})".format(round(coldwall_heatflux), expected_coldwall_heatflux))

    assert(round(coldwall_heatflux, 0) == expected_coldwall_heatflux)


if __name__ == '__main__':

    verify_default_simulation()
