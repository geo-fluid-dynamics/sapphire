"""A simulation class using an enthalpy-porosity method for binary alloys.

Use this for convection-coupled melting and solidification of phase-change materials with a single solute.
While application to metallic alloys is common, this can also be used e.g. for salt dissolved in water.

Dirichlet BC's should not be placed on the pressure.
The returned pressure solution will always have zero mean.

Non-homogeneous Neumann BC's are not implemented.

The PDE formulation (with a different discretization and solution procedure--also without regularization of the phase diagram--) was published in

    @article{parkinson2019jcpx,
        author = {Parkinson, James and Martin, Daniel and Wells, Andrew and Katz, Richard},
        year = {2019},
        month = {11},
        pages = {100043},
        title = {Modelling binary alloy solidification with Adaptive Mesh Refinement},
        volume = {5},
        journal = {Journal of Computational Physics: X},
        doi = {10.1016/j.jcpx.2019.100043}
    }
"""
import typing
import firedrake as fe
import sapphire.helpers
import sapphire.simulation
import sapphire.continuation
import sapphire.simulations.navier_stokes
import sapphire.simulations.enthalpy_porosity


DEFAULT_DISCRETIZATION_PARAMETERS = {
    'time_stencil_size': 2,
    'taylor_hood_pressure_element_degree': 1,
    'enthalpy_element_degree': 1,
    'solute_element_degree': 1,
    'quadrature_degree': 4,
}

DEFAULT_CONTINUATION_PARAMETERS = {
    'max_timestep_size_continuation_steps': 16,
    'max_porosity_smoothing_continuation_valu': 2.,
    'max_porosity_smoothing_continuation_steps': 8,
    'max_solute_rayleigh_number_continuation_steps': 16,
}

DEFAULT_SOLVER_PARAMETERS = sapphire.simulations.enthalpy_porosity.default_solver_parameters

sqrt = fe.sqrt

dot = fe.dot

inner = fe.inner

grad = fe.grad

div = fe.div

sym = fe.sym

erf = fe.erf


def phase_dependent_material_property(solid_to_liquid_ratio):

    a_sl = solid_to_liquid_ratio

    def a(phil):

        return a_sl + (1 - a_sl)*phil

    return a


def mushy_layer_porosity(S, H, c, p, r, Ste):

    a = r*(c - 1) + Ste*(p - 1)

    b = H*(1 - p) - Ste*p + r*(1 - 2*c) - S*(c - 1)

    c = H*p + c*(r + S)

    return (-b - sqrt(b**2 - 4*a*c))/(2*a)


def eutectic_porosity(S, r):

    return 1 + S/r


def eutectic_solid_porosity(H, Ste):

    return H/Ste


def solidus_enthalpy(S, c, p, r):

    assert p.__float__() >= 0.

    if p.__float__() == 0.:

        return fe.Constant(0.)

    else:

        return c*fe.Max(0., -1./p*(S + r))


def eutectic_enthalpy(S, c, p, r, Ste):

    return fe.Max(eutectic_porosity(S, r) * Ste, solidus_enthalpy(S, c, p, r))


def liquidus_enthalpy(S, Ste):

    return Ste - S


def element(cell, taylor_hood_pressure_degree, enthalpy_degree, solute_degree):

    return fe.MixedElement(
        fe.FiniteElement("P", cell, taylor_hood_pressure_degree),
        fe.VectorElement("P", cell, taylor_hood_pressure_degree + 1),
        fe.FiniteElement("P", cell, enthalpy_degree),
        fe.FiniteElement("P", cell, solute_degree))


class Simulation(sapphire.Simulation):

    def __init__(
            self,
            *args,
            enthalpy_porosity_parameters,
            binary_alloy_parameters,
            discretization_parameters,
            gravity_direction=(0, -1),
            continuation_parameters=None,
            solver_parameters=None,
            **kwargs):

        if discretization_parameters is None:

            discretization_parameters = DEFAULT_DISCRETIZATION_PARAMETERS

        if continuation_parameters is None:

            continuation_parameters = DEFAULT_CONTINUATION_PARAMETERS

        if solver_parameters is None:

            solver_parameters = DEFAULT_SOLVER_PARAMETERS

        if "solution" not in kwargs:

            mesh = kwargs["mesh"]

            del kwargs["mesh"]

            kwargs["solution"] = fe.Function(fe.FunctionSpace(
                mesh,
                element(
                    mesh.ufl_cell(),
                    discretization_parameters['taylor_hood_pressure_element_degree'],
                    discretization_parameters['enthalpy_degree'],
                    discretization_parameters['solute_degree']
                    )))

        self.prandtl_number = fe.Constant(enthalpy_porosity_parameters['prandtl_number'])

        self.heat_capacity_solid_to_liquid_ratio = fe.Constant(enthalpy_porosity_parameters['heat_capacity_solid_to_liquid_ratio'])

        self.thermal_conductivity_solid_to_liquid_ratio = fe.Constant(enthalpy_porosity_parameters['thermal_conductivity_solid_to_liquid_ratio'])

        self.stefan_number = fe.Constant(enthalpy_porosity_parameters['stefan_number'])

        self.porosity_smoothing_factor = fe.Constant(enthalpy_porosity_parameters['porosity_smoothing_factor'])

        self.partition_coefficient = fe.Constant(binary_alloy_parameters['partition_coefficient'])

        self.darcy_number = fe.Constant(binary_alloy_parameters['darcy_number'])

        self.lewis_number = fe.Constant(binary_alloy_parameters['lewis_number'])

        self.concentration_ratio = fe.Constant(binary_alloy_parameters['concentration_ratio'])

        self.solute_rayleigh_number = fe.Constant(binary_alloy_parameters['solute_rayleigh_number'])

        self.temperature_rayleigh_number = fe.Constant(binary_alloy_parameters['temperature_rayleigh_number'])

        self.reference_permeability = fe.Constant(binary_alloy_parameters['reference_permeability'])

        self.unit_gravity_direction = sapphire.helpers.normalize_to_unit_vector(gravity_direction)

        frame_translation_velocity = binary_alloy_parameters['frame_translation_velocity']
        
        if frame_translation_velocity is None:

            frame_translation_velocity = (0.,)*self.solution.function_space().mesh.geometric_dimension()

        self.frame_translation_velocity = fe.Constant(frame_translation_velocity)

        self.continuation_parameters = continuation_parameters

        self.porosity_smoothing_sequence = None

        sapphire.Simulation.__init__(
            self,
            *args,
            fieldnames=("p", "U", "S", "H"),
            discretization_parameters=discretization_parameters,
            continuation_parameters=continuation_parameters,
            solver_parameters=solver_parameters,
            **kwargs)

    def porosity(self, bulk_solute, enthalpy):

        S = bulk_solute

        H = enthalpy

        c = self.heat_capacity_solid_to_liquid_ratio

        p = self.partition_coefficient

        r = self.concentration_ratio

        Ste = self.stefan_number

        σ = self.porosity_smoothing_factor

        H_S = solidus_enthalpy(S, c, p, r)

        H_E = eutectic_enthalpy(S, c, p, r, Ste)

        H_L = liquidus_enthalpy(S, Ste)

        F_ES = eutectic_solid_porosity(H, Ste)

        F_ML = mushy_layer_porosity(S, H, c, p, r, Ste)

        return (1 + (F_ML - F_ES)*erf((H - H_E)/(sqrt(2)*σ)) - (F_ML - 1)*erf((H - H_L)/(sqrt(2)*σ)) + F_ES*erf((H - H_S)/(sqrt(2)*σ)))/2

    def temperature(self, bulk_solute, enthalpy):

        H = enthalpy

        phi = self.porosity(bulk_solute, enthalpy)

        Ste = self.stefan_number

        c = self.heat_capacity_solid_to_liquid_ratio

        return (H - Ste*phi)/(phi + (1 - phi)*c)

    def liquid_solute(self, bulk_solute, enthalpy):

        S = bulk_solute

        phi = self.porosity(bulk_solute, enthalpy)

        p = self.partition_coefficient

        return S/(phi + p*(1 - phi))  # Lever Rule

    def permeability(self, bulk_solute, enthalpy):

        phi = self.porosity(bulk_solute, enthalpy)

        Pi_c = self.reference_permeability

        # Eq. (8) from PMWK2019, "modified Carman-Kozeny permeability function appropriate to solidification in a thin Hele-Shaw cell"
        return (Pi_c**(-1) + (phi**3/(1 - phi)**2)**(-1))**(-1)

    def mass(self):

        U = self.solution_fields["U"]

        psi_p = self.test_functions["p"]

        dx = fe.dx(degree=self.quadrature_degree)

        return psi_p*div(U)*dx

    def momentum(self):

        Da = self.darcy_number

        Pr = self.prandtl_number

        Ra_T = self.temperature_rayleigh_number

        Ra_S = self.solute_rayleigh_number

        ghat = fe.Constant(sum([self.unit_vectors[i]*self.unit_gravity_direction[i]
                           for i in range(len(self.unit_vectors))]))

        p = self.solution_fields['p']

        U = self.solution_fields['U']

        S = self.solution_fields['S']

        H = self.solution_fields['H']

        U_t = self.time_discrete_terms['U']

        psi_U = self.test_functions['U']

        dx = fe.dx(degree=self.quadrature_degree)

        phi = self.porosity(S, H)

        T = self.temperature(S, H)

        S_L = self.liquid_solute(S, H)

        Pi = self.permeability(S, H)

        return (dot(psi_U, U_t + grad(U/phi)*U + Pr*phi*((Ra_T*T - Ra_S*S_L))*ghat + U/(Da*Pi)) - div(psi_U)*phi*p + Pr*inner(sym(grad(psi_U)), sym(grad(U))))*dx

    def energy(self):

        V_frame = self.frame_translation_velocity

        U = self.solution_fields['U']

        S = self.solution_fields['S']

        H = self.solution_fields['H']

        phi = self.porosity(S, H)

        T = self.temperature(S, H)

        k_sl = self.thermal_conductivity_solid_to_liquid_ratio

        k = phase_dependent_material_property(k_sl)(phi)

        psi_H = self.test_functions["H"]

        H_t = self.time_discrete_terms["H"]

        dx = fe.dx(degree=self.quadrature_degree)

        return (psi_H*(H_t + dot(V_frame, grad(H)) + dot(U, grad(T))) + dot(grad(psi_H), k*grad(T)))*dx

    def solute(self):

        Le = self.lewis_number

        V_frame = self.frame_translation_velocity

        U = self.solution_fields['U']

        S = self.solution_fields['S']

        H = self.solution_fields['H']

        phi = self.porosity(S, H)

        S_l = self.liquid_solute(S, H)

        psi_S = self.test_functions["S"]

        S_t = self.time_discrete_terms["S"]

        dx = fe.dx(degree=self.quadrature_degree)

        return (psi_S*(S_t + dot(V_frame, grad(S)) + dot(U, grad(S_l))) + 1./Le*dot(grad(psi_S), phi*grad(S_l)))*dx

    def weak_form_residual(self):

        return self.mass() + self.momentum() + self.energy() + self.solute()

    def solve(self):

        return sapphire.simulations.navier_stokes.Simulation.solve(self)

    def solve_with_porosity_over_smoothing(self):

        return sapphire.continuation.solve_with_over_regularization(
            solve=self.solve,
            solution=self.solution,
            regularization_parameter=self.porosity_smoothing_factor,
            maxval=self.max_porosity_smoothing_continuation_value,
            regularization_parameter_name="𝜎")

    def solve_with_bounded_porosity_smoothing(self):

        return sapphire.continuation.solve_with_bounded_regularization_sequence(
            solve=self.solve,
            solution=self.solution,
            backup_solution=self.backup_solution,
            regularization_parameter=self.porosity_smoothing_factor,
            initial_regularization_sequence=self.porosity_smoothing_sequence,
            maxcount=self.max_porosity_smoothing_continuation_steps,
            regularization_parameter_name="𝜎")

    def solve_with_porosity_smoothing_continuation(self):

        sigma = self.porosity_smoothing_factor.__float__()

        if self.porosity_smoothing_sequence is None:

            given_smoothing_sequence = False

        else:

            given_smoothing_sequence = True

        if not given_smoothing_sequence:
            # Find an over-regularization that works.
            self.solution, sigma_max = self.solve_with_porosity_over_smoothing()

            if sigma_max == sigma:
                # No over-regularization was necessary.
                return self.solution

            else:
                # A working over-regularization was found, which becomes
                # the upper bound of the sequence.
                self.phase_diagram_smoothing_sequence = (sigma_max, sigma)

        # At this point, either a smoothing sequence has been provided,
        # or a working upper bound has been found.
        # Next, a viable sequence will be sought.
        try:

            self.solution, self.porosity_smoothing_sequence = self.solve_with_bounded_porosity_smoothing()

        except fe.exceptions.ConvergenceError as error:

            if given_smoothing_sequence:
                # Try one more time without using the given sequence.
                # This is sometimes useful after solving some time steps
                # with a previously successful regularization sequence
                # that is not working for a new time step.
                self.solution = self.solution.assign(self.solutions[1])

                self.solution, smax = self.solve_with_phase_diagram_over_smoothing()

                self.smoothing_sequence = (smax, sigma)

                self.solution, self.smoothing_sequence = self.solve_with_bounded_phase_diagram_smoothing_sequence()

            else:

                raise error

        # For debugging purposes, ensure that the problem was solved with the
        # correct regularization and that the simulation's attribute for this
        # has been set to the correct value before returning.
        assert(self.phase_diagram_smoothing_factor.__float__() == sigma)

        return self.solution

    def nullspace(self):
        """Inform solver that pressure solution is not unique.

        It is only defined up to adding an arbitrary constant.
        """
        W = self.solution_space

        return fe.MixedVectorSpaceBasis(
            W,
            [fe.VectorSpaceBasis(constant=True),
             self.solution_subspaces['U'],
             self.solution_subspaces['S'],
             self.solution_subspaces['H']])

    def postprocess(self):

        _, U, S, H = self.solution.split()

        phi = fe.interpolate(self.porosity(S, H), S.function_space())

        self.post_processed_solutions['phi'] = phi

        self.velocity_divergence = fe.assemble(div(U)*self.dx)

        self.total_solute = fe.assemble(S*self.dx)

        self.total_energy = fe.assemble(H*self.dx)

        self.liquid_area = fe.assemble(phi*self.dx)

        self.max_speed = U.vector().array().max()

        self.minimum_porosity = phi.vector().array().min()

        self.maximum_porosity = phi.vector().array().max()

        self.minimum_solute = S.vector().array().min()

        self.maximum_solute = S.vector().array().max()

        return self

    def validate(self):

        tolerance = 1.e-2  # @todo Smaller tolerance wasn't working for phi so double check how that is being post-processed

        allowable_min_solute = -self.concentration_ratio.__float__()

        allowable_max_solute = 0.  # The nondimensional eutectic concentration is zero

        if not self.minimum_solute >= (allowable_min_solute - tolerance):

            raise Exception("Maximum bulk solute concentration {} is below allowable minimum of {}".format(
                self.minimum_solute, allowable_min_solute))

        if not self.maximum_solute <= (allowable_max_solute + tolerance):

            raise Exception("Minimum bulk solute concentration {} is above allowable maximum of {}".format(
                self.maximum_solute, allowable_max_solute))

        """
        if not self.minimum_porosity >= -tolerance:

            raise Exception("Minimum porosity {} is below minimum physically valid value of {}".format(
                self.minimum_porosity, 0.))

        if not self.maximum_porosity <= (1. + tolerance):

            raise Exception("Maximum porosity {} is above maximum physically valid value of {}".format(
                self.maximum_porosity, 1.))
        """

    def kwargs_for_writeplots(self):

        p = self.solution_subfunctions['p']

        U = self.solution_subfunctions['U']

        S = self.solution_subfunctions['S']

        H = self.solution_subfunctions['H']

        phi = self.post_processed_solutions['phi']

        return {
            'fields': (p, U, S, H, phi),
            'labels': ('p', '\\mathbf{U}', 'S', 'H', '\\phi'),
            'names': ('p', 'U', 'S', 'H', 'phi'),
            'plotfuns': (fe.tripcolor, fe.quiver, fe.tripcolor, fe.tripcolor, fe.tripcolor)}  # @todo Try fe.streamplot instead of fe.quiver here; but it threw a very weird error when I briefly tried

    def solve_with_solute_rayleigh_number_continuation(self) -> typing.Tuple[fe.Function, typing.Dict]:

        return sapphire.continuation.solve_with_bounded_regularization_sequence(
            solution=self.solution,
            # solve=self.solve_with_phase_diagram_smoothing_continuation,
            solve=self.solve,
            regularization_parameter=self.solute_rayleigh_number,
            initial_regularization_sequence=(0., self.solute_rayleigh_number.__float__()),
            regularization_parameter_name="solute_rayleigh_number",
            maxcount=self.max_solute_rayleigh_number_continuation_steps)

    def run_using_solute_rayleigh_number_continuation_for_first_timestep(self, *args, write_plots, **kwargs) -> typing.Tuple[typing.List[typing.Dict], typing.Dict]:

        self.postprocess()

        self.write_outputs(
            headers=True,
            checkpoint=True,
            vtk=False,
            plots=write_plots)

        if self.time.__float__() >= kwargs['endtime']:

            return

        # Solve first time step with continuation on the top wall boundary condition value because it is not consistent with the initial values.
        self.states = self.push_back_states()

        self.time = self.time.assign(self.time + self.timestep_size)

        self.state["index"] += 1

        self.solution, info = self.solve_with_solute_rayleigh_number_continuation()

        solution, _ = sapphire.Simulation.run(
            self,
            *args,
            write_plots=write_plots,
            **kwargs)

        return solution, info
