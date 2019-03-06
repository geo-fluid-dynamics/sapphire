""" Verify a FEM model via the Method of Manufactured Solution (MMS).

This module assumes that the FEM model is for a weak form
which approximates a strong form PDE.
We verify that the solved problem approximates the strong form.
"""
import firedrake as fe
import fempy.output
import fempy.table
import math
import matplotlib.pyplot as plt
import pathlib


def mms_source(
        model,
        strong_residual,
        manufactured_solution):
    
    V = model.solution.function_space()
    
    _r = strong_residual(
        model = model, solution = manufactured_solution(model))
        
    if type(model.element) is fe.FiniteElement:
    
        r = (_r,)
        
    else:
    
        r = _r
    
    psi = fe.TestFunctions(V)
    
    s = fe.inner(psi[0], r[0])
    
    if len(r) > 1:    
    
        for psi_i, r_i in zip(psi[1:], r[1:]):
            
            s += fe.inner(psi_i, r_i)
        
    return s
    
    
def mms_initial_values(model, manufactured_solution):

    initial_values = fe.Function(model.function_space)
    
    if type(model.element) is fe.FiniteElement:
    
        w_m = (manufactured_solution,)
        
    else:
    
        w_m = manufactured_solution
        
    for u_m, V in zip(
            w_m, model.function_space):
        
        initial_values.assign(fe.interpolate(u_m, V))
        
    return initial_values
    
    
def mms_dirichlet_boundary_conditions(model, manufactured_solution):
    
    W = model.function_space
    
    if type(model.element) is fe.FiniteElement:
    
        w = (manufactured_solution,)
    
    else:
    
        w = manufactured_solution
        
    return [fe.DirichletBC(V, g, "on_boundary") for V, g in zip(W, w)]
    
    
def L2_error(solution, true_solution, integration_measure):
    
    dx = integration_measure
    
    w_h = solution.split()
    
    if len(w_h) == 1:
        
        u_h = w_h[0]
        
        u = true_solution
        
        e = math.sqrt(fe.assemble(fe.inner(u_h - u, u_h - u)*dx))
        
    else:
        
        e = 0.
        
        for u_h, u in zip(
                w_h, true_solution):
            
            e += fe.assemble(fe.inner(u_h - u, u_h - u)*dx)

        e = math.sqrt(e)
        
    return e

    
def make_mms_verification_model_class(
        model_module,
        manufactured_solution):
    
    def initial_values(model):
        
        return mms_initial_values(
            model = model,
            manufactured_solution = manufactured_solution(model))
        
    def dirichlet_boundary_conditions(model):
    
        return mms_dirichlet_boundary_conditions(
            model = model,
            manufactured_solution = manufactured_solution(model))
        
    class MMSVerificationModel(model_module.Model):
            
        def __init__(self, *args, **kwargs):
            
            super().__init__(*args, 
                initial_values = initial_values,
                dirichlet_boundary_conditions = dirichlet_boundary_conditions,
                **kwargs)
                
            self.variational_form -= mms_source(
                    model = self,
                    strong_residual = model_module.strong_residual,
                    manufactured_solution = manufactured_solution)\
                *self.integration_measure
            
            self.problem, self.solver = self.reset_problem_and_solver()
            
            self.L2_error = None
        
    return MMSVerificationModel
    
    
def verify_spatial_order_of_accuracy(
        model_module,
        manufactured_solution,
        meshes,
        expected_order,
        tolerance,
        model_constructor_kwargs = {},
        parameters = {},
        timestep_size = 1.e32,
        endtime = 0.,
        starttime = 0.,
        plot_errors = False,
        plot_solution = False,
        report = False):
    
    MMSVerificationModel = make_mms_verification_model_class(
        model_module = model_module,
        manufactured_solution = manufactured_solution)
    
    table = fempy.table.Table(("h", "L2_error", "spatial_order"))
    
    print("")
    
    for mesh in meshes:
        
        h = mesh.cell_sizes((0.,)*mesh.geometric_dimension())
        
        model = MMSVerificationModel(mesh = mesh, **model_constructor_kwargs)
        
        model.output_directory_path = model.output_directory_path.joinpath(
            "mms_space_p{0}/".format(expected_order))

        model.output_directory_path = \
            model.output_directory_path.joinpath(
            "Deltat{0}".format(timestep_size))
        
        model.output_directory_path = model.output_directory_path.joinpath(
            "h{0}/".format(h))
        
        model = model.assign_parameters(parameters)
        
        model.time = model.time.assign(starttime)
        
        model.timestep_size = model.timestep_size.assign(timestep_size)
        
        model.solutions, model.time = model.run(
            endtime = endtime, plot = plot_solution, report = report)
        
        model.L2_error = L2_error(
            solution = model.solution,
            true_solution = manufactured_solution(model),
            integration_measure = model.integration_measure)
                
        table.append({
            "h": h,
            "L2_error": model.L2_error})
            
        if len(table) > 1:
        
            h, e = table.data["h"], table.data["L2_error"]

            log = math.log
            
            r = h[-2]/h[-1]
            
            order = log(e[-2]/e[-1])/log(r)
            
            table.data["spatial_order"][-1] = order
        
        print(str(table))
    
    print("Last observed spatial order of accuracy is {0}".format(order))
    
    if plot_errors:
    
        h, e = table.data["h"], table.data["L2_error"]
        
        plt.loglog(h, e, marker = "o")
        
        plt.xlabel(r"$h$")
        
        plt.ylabel(r"$\Vert u - u_h \Vert$")
        
        plt.axis("equal")
        
        plt.grid(True)
        
        filepath = model.output_directory_path.joinpath(
            "e_vs_h").with_suffix(".png")
        
        print("Writing plot to " + str(filepath))
        
        plt.savefig(str(filepath))
        
        plt.close()
        
    assert(abs(order - expected_order) < tolerance)
    
    
def verify_temporal_order_of_accuracy(
        model_module,
        manufactured_solution,
        mesh,
        timestep_sizes,
        endtime,
        expected_order,
        tolerance,
        model_constructor_kwargs = {},
        parameters = {},
        starttime = 0.,
        plot_errors = False,
        plot_solution = False,
        report = False):
    
    h = mesh.cell_sizes((0.,)*mesh.geometric_dimension())
    
    MMSVerificationModel = make_mms_verification_model_class(
        model_module = model_module,
        manufactured_solution = manufactured_solution)
    
    table = fempy.table.Table(("Delta_t", "L2_error", "temporal_order"))
    
    model = MMSVerificationModel(
        mesh = mesh, **model_constructor_kwargs)
    
    assert(len(model.solutions) > 1)
    
    basepath = model.output_directory_path
    
    model = model.assign_parameters(parameters)
    
    print("")
    
    for timestep_size in timestep_sizes:
        
        model.timestep_size = model.timestep_size.assign(timestep_size)
        
        model.output_directory_path = basepath.joinpath(
            "mms_time_q{0}/".format(expected_order))
        
        model.output_directory_path = model.output_directory_path.joinpath(
            "h{0}".format(h))
            
        model.output_directory_path = model.output_directory_path.joinpath(
            "Deltat{0}".format(timestep_size))
        
        model.time = model.time.assign(starttime)
        
        model.solutions = model.assign_initial_values_to_solutions()
        
        model.solutions, model.time = model.run(
            endtime = endtime, plot = plot_solution, report = report)
            
        model.L2_error = L2_error(
            solution = model.solution,
            true_solution = manufactured_solution(model),
            integration_measure = model.integration_measure)
            
        table.append({
            "Delta_t": timestep_size,
            "L2_error": model.L2_error})
            
        if len(table) > 1:
        
            Delta_t, e = table.data["Delta_t"], table.data["L2_error"]

            log = math.log
            
            r = Delta_t[-2]/Delta_t[-1]

            order = log(e[-2]/e[-1])/log(r)
    
            table.data["temporal_order"][-1] = order
        
        print(str(table))
        
    print("Last observed temporal order of accuracy is {0}".format(order))
    
    if plot_errors:
    
        Delta_t, e = table.data["Delta_t"], table.data["L2_error"]
        
        plt.loglog(Delta_t, e, marker = "o")
        
        plt.xlabel(r"$\Delta t$")
        
        plt.ylabel(r"$\Vert u - u_h \Vert$")
        
        plt.axis("equal")
        
        plt.grid(True)
        
        filepath = model.output_directory_path.joinpath(
            "e_vs_Delta_t").with_suffix(".png")
        
        print("Writing plot to " + str(filepath))
        
        plt.savefig(str(filepath))
        
        plt.close()
        
    assert(abs(order - expected_order) < tolerance)
    