""" Heat-driven cavity example """
import firedrake as fe
import sapphire.simulations.navier_stokes_boussinesq
import sapphire.continuation


class Simulation(sapphire.simulations.navier_stokes_boussinesq.Simulation):
    """ Heat driven cavity example class derived from the Navier-Stokes-Boussinesq simulation class """

    def __init__(
            self,
            *args,
            mesh_dimensions=(20, 20),
            hotwall_temperature=0.5,
            coldwall_temperature=-0.5,
            reynolds_number=1.,
            rayleigh_number=1.e6,
            prandtl_number=0.71,
            **kwargs):

        if "solution" not in kwargs:

            kwargs["mesh"] = fe.UnitSquareMesh(*mesh_dimensions)

        self.hotwall_id = 1

        self.coldwall_id = 2

        self.hotwall_temperature = fe.Constant(hotwall_temperature)

        self.coldwall_temperature = fe.Constant(coldwall_temperature)

        super().__init__(
            *args,
            reynolds_number=reynolds_number,
            rayleigh_number=rayleigh_number,
            prandtl_number=prandtl_number,
            **kwargs)

    def dirichlet_boundary_conditions(self):

        d = self.solution.function_space().mesh().geometric_dimension()

        return [
            fe.DirichletBC(
                self.solution_subspaces["u"],
                (0,)*d,
                "on_boundary"),
            fe.DirichletBC(
                self.solution_subspaces["T"],
                self.hotwall_temperature,
                self.hotwall_id),
            fe.DirichletBC(
                self.solution_subspaces["T"],
                self.coldwall_temperature,
                self.coldwall_id)]

    def solve_with_rayleigh_number_continuation(self):

        self.solver_parameters['snex_max_it'] = 10

        self.solution, _ = sapphire.continuation.solve_with_bounded_regularization_sequence(
            solve=super().solve,
            solution=self.solution,
            regularization_parameter=self.rayleigh_number,
            initial_regularization_sequence=(1, self.rayleigh_number.__float__()))

        return self.solution
