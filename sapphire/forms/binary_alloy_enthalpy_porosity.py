from logging import raiseExceptions
from typing import Tuple, Callable, Any
from pathlib import Path
import cmocean
from matplotlib import ticker
from matplotlib import pyplot as plt
import numpy
import scipy.special
from firedrake import sqrt as firedrake_sqrt
from firedrake import erf as firedrake_erf
from firedrake import Constant, Cell, MixedElement, FiniteElement, VectorElement, dx, div, inner, grad, dot, sym, interpolate, assemble, FunctionSpace, SpatialCoordinate, tricontourf
from firedrake.utility_meshes import UnitSquareMesh
from sapphire.data.solution import Solution
from sapphire.time_discretization import ufl_time_discrete_terms
from sapphire.forms.incompressible_flow import mass_residual
from sapphire.io.plot import DEFAULT_SAVEFIG_KWARGS


COMPONENT_NAMES = ('p', 'U', 'S', 'H')

# Constants arising from nondimensionalization
#
# T = (T_dim - T_e_dim)/(T_L(S_0_dim) - T_e_dim)
EUTECTIC_TEMPERATURE = 0.

# H = Ste * phi + (phi + (1 - phi)c_sl)*T
EUTECTIC_ENTHALPY = 0.

# S = (S_dim - S_e_dim)/(S_e_dim - S_0_dim)
EUTECTIC_CONCENTRATION = 0.

INITIAL_SOLUTE_CONCENTRATION = -1

SOLIDUS_ENTHALPY = 0.  # Assuming zero partition coefficient
#


def element(cell: Cell, solute_degree: int, enthalpy_degree: int, taylor_hood_velocity_degree: int):

    if taylor_hood_velocity_degree < 2:

        raise Exception("Taylor-Hood velocity element degree must be at least 2")

    if solute_degree < 1:

        raise Exception("Solute element degree must be at least 1 because continuous Galerkin discretization is assumed")

    if enthalpy_degree < 1:

        raise Exception("Enthalpy element degree must be at least 1 because continuous Galerkin discretization is assumed")

    return MixedElement(
        FiniteElement('P', cell, taylor_hood_velocity_degree - 1),
        VectorElement('P', cell, taylor_hood_velocity_degree),
        FiniteElement('P', cell, solute_degree),
        FiniteElement('P', cell, enthalpy_degree))


def phase_dependent_material_property(solid_to_liquid_ratio):

    a_sl = solid_to_liquid_ratio

    def a(phil):

        return a_sl + (1 - a_sl)*phil

    return a


def _mushy_layer_porosity(S, H, r, c_sl, Ste):

    a = r*(c_sl - 1) - Ste

    b = H + r*(1 - 2*c_sl) - S*(c_sl - 1)

    c = c_sl*(r + S)

    if isinstance(S, numpy.ndarray):

        sqrt = numpy.sqrt

    else:

        sqrt = firedrake_sqrt

    return (-b - sqrt(b**2 - 4*a*c))/(2*a)


def _eutectic_porosity(S, r):

    return 1 + S/r


def _eutectic_solid_porosity(H, Ste):

    return H/Ste


def _eutectic_enthalpy(S, r, Ste):

    return _eutectic_porosity(S=S, r=r) * Ste


def _liquidus_enthalpy(S, Ste):

    return Ste - S


def __exact_porosity(S, H, c_sl, r, Ste):

    f_array = numpy.full_like(S, -1)

    F_S = 0.

    F_L = 1.

    H_S = SOLIDUS_ENTHALPY

    H_E = _eutectic_enthalpy(S=S, r=r, Ste=Ste)

    H_L = _liquidus_enthalpy(S=S, Ste=Ste)

    F_ES = _eutectic_solid_porosity(H=H, Ste=Ste)

    F_ML = _mushy_layer_porosity(S=S, H=H, r=r, c_sl=c_sl, Ste=Ste)

    f_array = numpy.where(H <= H_S, F_S, f_array)

    f_array = numpy.where((H_S < H) & (H <= H_E), F_ES, f_array)

    f_array = numpy.where((H_E < H) & (H < H_L), F_ML, f_array)

    f_array = numpy.where(H_L <= H, F_L, f_array)

    return f_array


def _porosity(S, H, c_sl, r, Ste, σ):

    H_S = SOLIDUS_ENTHALPY

    H_E = _eutectic_enthalpy(S=S, r=r, Ste=Ste)

    H_L = _liquidus_enthalpy(S=S, Ste=Ste)

    F_ES = _eutectic_solid_porosity(H=H, Ste=Ste)

    F_ML = _mushy_layer_porosity(S=S, H=H, r=r, c_sl=c_sl, Ste=Ste)

    if isinstance(S, numpy.ndarray):

        erf = scipy.special.erf

        sqrt = numpy.sqrt

    else:

        erf = firedrake_erf

        sqrt = firedrake_sqrt

    return (1 + (F_ML - F_ES)*erf((H - H_E)/(sqrt(2)*σ)) - (F_ML - 1)*erf((H - H_L)/(sqrt(2)*σ)) + F_ES*erf((H - H_S)/(sqrt(2)*σ)))/2


def porosity(solution: Solution):

    return _porosity(
        S=solution.ufl_fields.S,
        H=solution.ufl_fields.H,
        c_sl=solution.ufl_constants.heat_capacity_solid_to_liquid_ratio,
        r=solution.ufl_constants.concentration_ratio,
        Ste=solution.ufl_constants.stefan_number,
        σ=solution.ufl_constants.porosity_smoothing_factor)


def temperature(solution: Solution):

    H = solution.ufl_fields.H

    phi = porosity(solution)

    Ste = solution.ufl_constants.stefan_number

    c_sl = solution.ufl_constants.heat_capacity_solid_to_liquid_ratio

    return (H - Ste*phi)/(phi + (1 - phi)*c_sl)


def liquid_solute(solution: Solution):

    S = solution.ufl_fields.S

    phi = porosity(solution)

    # S = phi*S_l + (1 - phi) * S_s (Lever Rule)
    return S/phi


def linear_boussinesq_buoyancy(solution: Solution):

    T = temperature(solution)

    S_l = liquid_solute(solution)

    Ra_T = solution.ufl_constants.temperature_rayleigh_number

    Ra_S = solution.ufl_constants.solute_rayleigh_number

    return Ra_T*T - Ra_S*S_l


def thin_hele_shaw_cell_permeability(solution: Solution):

    phi = porosity(solution)

    Pi_c = solution.ufl_constants.reference_permeability

    # Eq. (8) from PMWK2019, "modified Carman-Kozeny permeability function appropriate to solidification in a thin Hele-Shaw cell"
    return (Pi_c**(-1) + (phi**3/(1 - phi)**2)**(-1))**(-1)


def normalized_kozeny_carman_permeability(solution: Solution):
    # Pi/Pi_0 from Le Bars 2006, where Pi is the permeability and Pi_0 is an empirically calibrated reference permeability.
    # Note that Pi_0 appears again in the Darcy number.
    phi = porosity(solution)

    return phi**3/(1 - phi)**2


def momentum_residual(solutions: Tuple[Solution], buoyancy: Callable[[Solution], Any], permeability: Callable[[Solution], Any]):

    solution = solutions[0]

    Da = solution.ufl_constants.darcy_number

    Pr = solution.ufl_constants.prandtl_number

    p = solution.ufl_fields.p

    U = solution.ufl_fields.U

    U_t = ufl_time_discrete_terms(solutions).U_t  # pylint: disable=no-member

    psi_U = solution.test_functions.U

    _dx = dx(degree=solution.quadrature_degree)

    phi = porosity(solution)

    b = buoyancy(solution)

    ghat = Constant(-solution.unit_vectors[1])

    Pi = permeability(solution)

    return (dot(psi_U, U_t + grad(U/phi)*U + Pr*phi*(b*ghat + U/(Da*Pi))) - div(psi_U)*phi*p + Pr*inner(sym(grad(psi_U)), sym(grad(U))))*_dx


def energy_residual(solutions: Tuple[Solution]):

    solution = solutions[0]

    V_frame = solution.ufl_constants.frame_translation_velocity

    U = solution.ufl_fields.U

    H = solution.ufl_fields.H

    phi = porosity(solution)

    T = temperature(solution)

    k_sl = solution.ufl_constants.thermal_conductivity_solid_to_liquid_ratio

    k = phase_dependent_material_property(k_sl)(phi)

    H_t = ufl_time_discrete_terms(solutions).H_t  # pylint: disable=no-member

    psi_H = solution.test_functions.H

    _dx = dx(degree=solution.quadrature_degree)

    return (psi_H*(H_t + dot(V_frame, grad(H)) + dot(U, grad(T))) + dot(grad(psi_H), k*grad(T)))*_dx


def solute_residual(solutions: Tuple[Solution]):

    solution = solutions[0]

    Le = solution.ufl_constants.lewis_number

    V_frame = solution.ufl_constants.frame_translation_velocity

    U = solution.ufl_fields.U

    S = solution.ufl_fields.S

    phi = porosity(solution)

    S_l = liquid_solute(solution)

    psi_S = solution.test_functions.S

    S_t = ufl_time_discrete_terms(solutions).S_t  # pylint: disable=no-member

    _dx = dx(degree=solution.quadrature_degree)

    return (psi_S*(S_t + dot(V_frame, grad(S)) + dot(U, grad(S_l))) + 1./Le*dot(grad(psi_S), phi*grad(S_l)))*_dx


def residual(solutions: Solution, buoyancy=linear_boussinesq_buoyancy, permeability=normalized_kozeny_carman_permeability):

    return mass_residual(solutions) + momentum_residual(solutions, buoyancy=buoyancy, permeability=permeability) + energy_residual(solutions) + solute_residual(solutions)


def postprocess(solution: Solution):

    U = solution.subfunctions.U

    S = solution.subfunctions.S

    H = solution.subfunctions.H

    phi = interpolate(porosity(solution), S.function_space())

    _dx = dx(degree=solution.quadrature_degree)

    return {
        '\\phi': phi,
        'velocity_divergence': assemble(div(U)*_dx),
        'total_solute': assemble(S*_dx),
        'total_energy': assemble(H*_dx),
        'liquid_area': assemble(phi*_dx),
        'max_speed': U.vector().array().max(),
        'minimum_porosity': phi.vector().array().min(),
        'maximum_porosity': phi.vector().array().max(),
        'minimum_solute': S.vector().array().min(),
        'maximum_solute': S.vector().array().max(),
        'minimum_enthalpy': H.vector().array().min(),
        'maximum_enthalpy': H.vector().array().max()}


def _minimum_allowable_solute_concentration(r):

    return -r


def validate(solution: Solution):

    tolerance = 0.05  # @todo Smaller tolerance wasn't working for phi so double check how that is being post-processed

    allowable_min_solute = _minimum_allowable_solute_concentration(solution.ufl_constants.concentration_ratio.__float__())

    allowable_max_solute = 0.  # The nondimensional eutectic concentration is zero

    minimum_enthalpy = solution.post_processed_objects['minimum_enthalpy']

    if (minimum_enthalpy < EUTECTIC_ENTHALPY):

        raise Exception("Minimum enthalpy {} is below allowable minimum of {}".format(minimum_enthalpy, EUTECTIC_ENTHALPY))

    minimum_solute = solution.post_processed_objects['minimum_solute']

    if minimum_solute < (allowable_min_solute - tolerance):

        raise Exception("Minimum bulk solute concentration {} is below allowable minimum of {}".format(minimum_solute, allowable_min_solute))

    maximum_solute = solution.post_processed_objects['maximum_solute']

    if maximum_solute > (allowable_max_solute + tolerance):

        raise Exception("Maximum bulk solute concentration {} is above allowable maximum of {}".format(maximum_solute, allowable_max_solute))

    minimum_porosity = solution.post_processed_objects['minimum_porosity']

    if minimum_porosity < -tolerance:

        raise Exception("Minimum porosity {} is below minimum physically valid value of {}".format(minimum_porosity, 0.))

    maximum_porosity = solution.post_processed_objects['maximum_porosity']

    if maximum_porosity > (1. + tolerance):

        raise Exception("Maximum porosity {} is above maximum physically valid value of {}".format(maximum_porosity, 1.))


def _phase_diagram_mesh(
        S_min,
        H_max,
        resolution):

    return numpy.meshgrid(numpy.linspace(S_min, EUTECTIC_CONCENTRATION, resolution), numpy.linspace(SOLIDUS_ENTHALPY, H_max, resolution))


def plot_phase_diagram(
        fig,
        axes,
        S_min,
        H_max,
        z_mesh,
        z_label,
        colormap,
        fontsize=11,
        x_tick_string_format="%.2f",
        y_tick_string_format="%.2f",
        S_tick_count=5,
        H_tick_count=5,
        imshow_kwargs=None):

    if imshow_kwargs is None:

        imshow_kwargs = {}

    image = axes.imshow(
        z_mesh,
        extent=(S_min, EUTECTIC_CONCENTRATION, SOLIDUS_ENTHALPY, H_max),
        origin="lower",
        cmap=colormap,
        interpolation="bilinear",
        **imshow_kwargs)

    S_ticks = numpy.linspace(S_min, EUTECTIC_CONCENTRATION, S_tick_count)

    axes.set_xticks(S_ticks)

    H_ticks = numpy.linspace(EUTECTIC_ENTHALPY, H_max, H_tick_count)

    axes.set_yticks(H_ticks)

    axes.xaxis.set_major_formatter(ticker.FormatStrFormatter(x_tick_string_format))

    axes.yaxis.set_major_formatter(ticker.FormatStrFormatter(y_tick_string_format))

    axes.set_aspect((EUTECTIC_CONCENTRATION - S_min) / (H_max - EUTECTIC_ENTHALPY))

    colorbar_axes = fig.add_axes([axes.get_position().x1 + 0.01, axes.get_position().y0, 0.01, axes.get_position().height])

    colorbar = plt.colorbar(image, cax=colorbar_axes)

    colorbar.ax.set_title("${}$ [-]".format(z_label), fontsize=fontsize)

    return image, colorbar


def plot_phase_diagram_with_and_without_smoothing(
        heat_capacity_solid_to_liquid_ratio: float,
        concentration_ratio: float,
        stefan_number: float,
        porosity_smoothing_factor: float,
        output_directory_path: str,
        resolution: int = 256,
        fontsize=11,
        H_top_margin=0.1,
        fig_basesize=4,
        savefig_kwargs=None):

    if savefig_kwargs is None:

        savefig_kwargs = DEFAULT_SAVEFIG_KWARGS

    c_sl = heat_capacity_solid_to_liquid_ratio

    r = concentration_ratio

    Ste = stefan_number

    σ = porosity_smoothing_factor

    S_min = -r

    H_max = (1 + H_top_margin)*(Ste - S_min)

    S_mesh, H_mesh = _phase_diagram_mesh(
        S_min=S_min,
        H_max=H_max,
        resolution=resolution)

    S_array = numpy.ravel(S_mesh)

    H_array = numpy.ravel(H_mesh)

    f_array = __exact_porosity(
        S=S_array,
        H=H_array,
        c_sl=c_sl,
        r=r,
        Ste=Ste)

    f_mesh = f_array.reshape(S_mesh.shape)

    phi_array = _porosity(
        S=S_array,
        H=H_array,
        c_sl=c_sl,
        r=r,
        Ste=Ste,
        σ=σ)

    phi_mesh = phi_array.reshape(S_mesh.shape)

    e_array = abs(phi_array - f_array)

    e_mesh = e_array.reshape(S_mesh.shape)

    ncols = 3

    fig, (f_axes, phi_axes, e_axes) = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols*fig_basesize, fig_basesize), sharex=False, sharey=True)

    plt.subplots_adjust(wspace=0.35)

    plot_phase_diagram(fig=fig, axes=f_axes, S_min=S_min, H_max=H_max, z_mesh=f_mesh, z_label="f", colormap=cmocean.cm.ice_r)

    f_axes.set_xlabel(r"$S$ [-]", fontsize=fontsize)

    f_axes.set_ylabel(r"$H$ [-]", fontsize=fontsize)

    plot_phase_diagram(fig=fig, axes=phi_axes, S_min=S_min, H_max=H_max, z_mesh=phi_mesh, z_label=r"\phi", colormap=cmocean.cm.ice_r)

    phi_axes.set_xlabel(r"$S$ [-]", fontsize=fontsize)

    plot_phase_diagram(fig=fig, axes=e_axes, S_min=S_min, H_max=H_max, z_mesh=e_mesh, z_label=r"\left| \phi - f \right|", colormap=plt.cm.hot_r, imshow_kwargs={'vmin': 0., 'vmax': 1.})

    e_axes.set_xlabel(r"$S$ [-]", fontsize=fontsize)

    filepath = output_directory_path + '/f_phi_e__csl{}_r{}_Ste{}_sigma{}.png'.format(c_sl, r, Ste, σ)

    print("Writing plot to {}".format(filepath))

    fig.savefig(filepath, **savefig_kwargs)


def plot_phase_diagram_using_firedrake(
        concentration_ratio: float,
        heat_capacity_solid_to_liquid_ratio: float,
        stefan_number: float,
        porosity_smoothing_factor: float,
        output_directory_path: str,
        H_top_margin: float = 0.1,
        savefig_kwargs: dict = None,
        resolution: int = 256):

    c_sl = heat_capacity_solid_to_liquid_ratio

    r = concentration_ratio

    Ste = stefan_number

    σ = porosity_smoothing_factor

    S_min = _minimum_allowable_solute_concentration(r=r)

    S_max = EUTECTIC_CONCENTRATION

    H_min = SOLIDUS_ENTHALPY

    H_max = (1 + H_top_margin)*(Ste - S_min)

    phase_diagram_mesh = UnitSquareMesh(resolution, resolution)

    x = SpatialCoordinate(phase_diagram_mesh)

    phi = interpolate(
        _porosity(
            S=S_min + (S_max - S_min)*x[0],
            H=H_min + (H_max - H_min)*x[1],
            c_sl=c_sl,
            r=r,
            Ste=Ste,
            σ=σ),
        FunctionSpace(phase_diagram_mesh, FiniteElement('P', phase_diagram_mesh.ufl_cell(), 1)))

    Path(output_directory_path).mkdir(parents=True, exist_ok=True)

    if savefig_kwargs is None:

        savefig_kwargs = DEFAULT_SAVEFIG_KWARGS

    _, axes = plt.subplots()

    mappable = tricontourf(phi, axes=axes, levels=128, cmap=cmocean.cm.ice_r)

    axes.set_aspect(1.)

    plt.colorbar(mappable=mappable, ax=axes)

    plt.title(r'$\phi$')

    plt.xlabel('$S$ normalized')

    plt.ylabel('$H$ normalized')

    filepath = output_directory_path + '/phase_diagram_with_smoothing_firedrake__csl{}_r{}_Ste{}_sigma{}.png'.format(c_sl, r, Ste, σ)

    print("Writing plot to {}".format(filepath))

    plt.savefig(filepath, **savefig_kwargs)

    plt.close()
