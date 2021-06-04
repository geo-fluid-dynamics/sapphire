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
import sapphire
import sapphire.examples.heat_driven_cavity
from sapphire.examples.heat_driven_cavity import mesh, element, nullspace, dirichlet_boundary_conditions, solve
import firedrake as fe


SOLUTION_FUNCTION_COMPONENT_NAMES = sapphire.examples.heat_driven_cavity.SOLUTION_FUNCTION_COMPONENT_NAMES

MESH_HOTWALL_ID = sapphire.examples.heat_driven_cavity.MESH_HOTWALL_ID

MESH_COLDWALL_ID = sapphire.examples.heat_driven_cavity.MESH_COLDWALL_ID


def ufl_constants_for_water_buoyancy(reference_temperature_range__degC: float):

    return {'reference_temperature_range__degC': reference_temperature_range__degC}


def buoyancy_with_density_anomaly_of_water(solution: sapphire.Solution):
    """Eq. (25) from @cite{danaila2014newton}"""
    T = solution.ufl_fields.T

    Re = solution.ufl_constants.reynolds_number

    Ra = solution.ufl_constants.rayleigh_number

    Pr = solution.ufl_constants.prandtl_number

    DeltaT = solution.ufl_constants.reference_temperature_range__degC

    ghat = fe.Constant(-solution.unit_vectors[1])

    T_anomaly_degC = fe.Constant(4.0293)

    rho_anomaly_SI = fe.Constant(999.972)

    w_degC = fe.Constant(9.2793e-6)

    q = fe.Constant(1.894816)

    def T_degC(T):
        """ T = T_degC/DeltaT """
        return DeltaT*T

    def rho_of_T_degC(T_degC):
        """ Eq. (24) from @cite{danaila2014newton} """
        return rho_anomaly_SI*(1. - w_degC*abs(T_degC - T_anomaly_degC)**q)

    def rho(T):

        return rho_of_T_degC(T_degC(T))

    beta = fe.Constant(6.91e-5)  # [K^-1]

    rho_0 = rho(T=0.)

    return Ra/(Pr*Re**2*beta*DeltaT)*(rho_0 - rho(T))/rho_0*ghat


def residual(solution: sapphire.Solution):

    return sapphire.examples.heat_driven_cavity.residual(solution, buoyancy=buoyancy_with_density_anomaly_of_water)


def output(solution: sapphire.Solution, outdir_path="sapphire_output/heat_driven_cavity_with_water/"):

    sapphire.examples.heat_driven_cavity.output(solution, outdir_path=outdir_path)


def run_simulation(
        reynolds_number=1.,
        rayleigh_number=2.518084e6,
        prandtl_number=6.99,
        hotwall_temperature=1.,
        coldwall_temperature=0.,
        reference_temperature_range__degC=10.,
        mesh_dimensions=(20, 20),
        taylor_hood_pressure_element_degree=1,
        temperature_element_degree=2
        ) -> sapphire.Simulation:

    _ufl_constants = {
        **sapphire.examples.heat_driven_cavity.ufl_constants(
            hotwall_temperature=hotwall_temperature,
            coldwall_temperature=coldwall_temperature,
            reynolds_number=reynolds_number,
            rayleigh_number=rayleigh_number,
            prandtl_number=prandtl_number),
        **ufl_constants_for_water_buoyancy(reference_temperature_range__degC=reference_temperature_range__degC)}

    _mesh = mesh(mesh_dimensions)

    _element = element(_mesh.ufl_cell(), taylor_hood_pressure_element_degree, temperature_element_degree)

    solution_function_space = fe.FunctionSpace(_mesh, _element)

    solution_function = fe.Function(solution_function_space)

    solution = sapphire.data.Solution(
        function=solution_function,
        function_component_names=SOLUTION_FUNCTION_COMPONENT_NAMES,
        ufl_constants=_ufl_constants)

    problem = sapphire.data.Problem(
        solution=solution,
        residual=residual,
        dirichlet_boundary_conditions=dirichlet_boundary_conditions(solution),
        nullspace=nullspace(solution))

    sim = sapphire.data.Simulation(problem=problem, solutions=(solution,))

    sim = sapphire.run(sim=sim, solve=solve, output=output)

    return sim


def verify_default_simulation():

    sim = run_simulation()

    solution = sim.solutions[0]

    ds = fe.ds(subdomain_id=MESH_COLDWALL_ID)

    nhat = fe.FacetNormal(solution.mesh)

    T = solution.ufl_fields.T

    dot, grad = fe.dot, fe.grad

    coldwall_heatflux = fe.assemble(dot(grad(T), nhat)*ds)

    expected_coldwall_heatflux = -8

    print("Integrated cold wall heat flux = {} (Expected {})".format(round(coldwall_heatflux), expected_coldwall_heatflux))

    assert(round(coldwall_heatflux, 0) == expected_coldwall_heatflux)


if __name__ == '__main__':

    verify_default_simulation()
