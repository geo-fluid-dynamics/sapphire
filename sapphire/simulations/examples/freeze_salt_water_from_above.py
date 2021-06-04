""" Example simulation for freezing salt water from above """
import typing
import sapphire.simulations.binary_alloy_enthalpy_porosity
import firedrake as fe


walls = {'bottom': 1, 'top': 2}

def solve_with_timestep_size_continuation(simulation: Simulation) -> fe.Function:

    return solve_with_bounded_regularization_sequence(
        solution=self.solution,
        solve=sapphire.Simulation.solve,
        regularization_parameter=self.form_parameters.timestep_size,
        initial_regularization_sequence=(0., self.form_parameters.timestep_size.__float__()),
        regularization_parameter_name="timestep_size",
        maxcount=self.continuation_parameters['max_timestep_size_continuation_steps'],
        start_from_right=True)


class Simulation(sapphire.simulations.binary_alloy_enthalpy_porosity.Simulation):

    def __init__(
            self,
            *args,
            nx,
            ny,
            Lx,
            Ly,
            initial_solute_concentration,
            initial_enthalpy,
            top_wall_enthalpy,
            max_top_bc_continuation_steps=8,
            **kwargs):

        if "solution" not in kwargs:

            kwargs["mesh"] = fe.PeriodicRectangleMesh(nx, ny, Lx, Ly, direction="x")

        self.initial_solute_concentration = fe.Constant(initial_solute_concentration)

        self.initial_enthalpy = fe.Constant(initial_enthalpy)

        self.top_wall_enthalpy = fe.Constant(top_wall_enthalpy)

        self.max_top_bc_continuation_steps = max_top_bc_continuation_steps

        sapphire.simulations.binary_alloy_enthalpy_porosity.Simulation.__init__(self, *args, **kwargs)

    def initial_values(self):

        w_0 = fe.Function(self.solution.function_space())

        p, U, S, H = w_0.split()

        p = p.assign(0.)

        ihat, jhat = self.unit_vectors

        U = U.assign(0.*ihat + 0.*jhat)

        S = S.assign(self.initial_solute_concentration)

        H = H.assign(self.initial_enthalpy)

        return w_0

    def dirichlet_boundary_conditions(self):

        return (
            fe.DirichletBC(self.solution_subspaces["U"], (0, 0), (walls['top'], walls['bottom'])),
            fe.DirichletBC(self.solution_subspaces["H"], self.top_wall_enthalpy, walls['top']),
            fe.DirichletBC(self.solution_subspaces["H"], self.initial_enthalpy, walls['bottom']),
        )

    def solve_with_top_wall_enthalpy_and_timestep_size_continuation(self) -> typing.Tuple[fe.Function, typing.Dict]:

        return sapphire.continuation.solve_with_bounded_regularization_sequence(
            solution=self.solution,
            solve=self.solve_with_timestep_size_continuation,
            regularization_parameter=self.top_wall_enthalpy,
            initial_regularization_sequence=(self.initial_enthalpy.__float__(), self.top_wall_enthalpy.__float__()),
            regularization_parameter_name="H_top",
            maxcount=self.max_top_bc_continuation_steps)

    def run_with_multi_parameter_continuation_for_first_timestep(self, *args, write_plots, **kwargs) -> typing.Tuple[typing.List[typing.Dict], typing.Dict]:

        self.postprocess()

        self.write_outputs(
            headers=True,
            checkpoint=True,
            vtk=False,
            plots=write_plots)

        if self.time.__float__() >= kwargs['endtime']:

            return {}

        # Solve first time step with continuation on the top wall boundary condition value because it is not consistent with the initial values.
        self.states = self.push_back_states()

        self.time = self.time.assign(self.time + self.timestep_size)

        self.state["index"] += 1

        self.solution, _ = self.solve_with_top_wall_enthalpy_and_timestep_size_continuation()

        # Solve the remaining time steps with only timestep size continuation.
        return sapphire.Simulation.run(
            *args,
            solve=sapphire.Simulation.solve_with_timestep_size_continuation,
            write_plots=write_plots,
            **kwargs)
