""" Verify a Simulation via the Method of Manufactured Solution (MMS).

This module assumes that the solution to the 
simulation's governing equations,
written as a weak form residual and defined using UFL,
approximate solutions to a strong form residual, 
which must also be defined using UFL.
"""
import firedrake as fe
import sapphire.output
import math
import pathlib
import pandas


def mms_source(
        sim,
        strong_residual,
        manufactured_solution):
    
    V = sim.solution.function_space()
    
    _r = strong_residual(sim, solution = manufactured_solution(sim))
    
    if type(sim.solution.function_space().ufl_element()) is fe.FiniteElement:
    
        r = (_r,)
        
    else:
    
        r = _r
    
    psi = fe.TestFunctions(V)
    
    s = fe.inner(psi[0], r[0])
    
    if len(r) > 1:    
    
        for psi_i, r_i in zip(psi[1:], r[1:]):
            
            s += fe.inner(psi_i, r_i)
        
    return s
    
    
def mms_initial_values(sim, manufactured_solution):
    
    initial_values = fe.Function(sim.solution_space)
    
    if type(sim.solution_space.ufl_element()) is fe.FiniteElement:
    
        w_m = (manufactured_solution,)
        
    else:
    
        w_m = manufactured_solution
        
    for iv, w_mi, W_i in zip(
            initial_values.split(), w_m, sim.solution_space):
        
        iv.assign(fe.interpolate(w_mi, W_i))
    
    return initial_values
    
    
def default_mms_dirichlet_boundary_conditions(sim, manufactured_solution):
    """Apply Dirichlet BC's to every component on every boundary."""
    W = sim.solution_space
    
    if type(W.ufl_element()) is fe.FiniteElement:
    
        w = (manufactured_solution,)
    
    else:
    
        w = manufactured_solution
        
    return [fe.DirichletBC(V, g, "on_boundary") for V, g in zip(W, w)]
    
    
def make_mms_verification_sim_class(
        Simulation,
        manufactured_solution,
        strong_residual,
        mms_dirichlet_boundary_conditions = None,
        write_simulation_outputs = False):
    
    if strong_residual is None:
        
        strong_residual = Simulation.strong_residual
    
    if mms_dirichlet_boundary_conditions is None:
    
        mms_dirichlet_boundary_conditions = \
            default_mms_dirichlet_boundary_conditions
    
    class MMSVerificationSimulation(Simulation):
        
        def weak_form_residual(self):
        
            return super().weak_form_residual() \
                - mms_source(
                    sim = self,
                    strong_residual = strong_residual,
                    manufactured_solution = manufactured_solution)\
                *fe.dx(degree = self.quadrature_degree)
        
        def initial_values(self):
        
            return mms_initial_values(
                sim = self,
                manufactured_solution = manufactured_solution(self))
        
        def dirichlet_boundary_conditions(self):
        
            return mms_dirichlet_boundary_conditions(
                sim = self,
                manufactured_solution = manufactured_solution(self))
        
        if not write_simulation_outputs:
        
            def write_outputs(self, *args, **kwargs):
            
                pass
    
    return MMSVerificationSimulation
    
    
def verify_spatial_order_of_accuracy(
        Simulation,
        manufactured_solution,
        meshes,
        norms,
        expected_orders = None,
        decimal_places = 2,
        time_dependent = True,
        sim_kwargs = {},
        endtime = 0.,
        strong_residual = None,
        dirichlet_boundary_conditions = None,
        starttime = 0.,
        outfile = None,
        write_simulation_outputs = False):
    
    
    MMSVerificationSimulation = make_mms_verification_sim_class(
        Simulation = Simulation,
        manufactured_solution = manufactured_solution,
        write_simulation_outputs = write_simulation_outputs,
        strong_residual = strong_residual,
        mms_dirichlet_boundary_conditions = dirichlet_boundary_conditions)
    
    
    fieldcount = len(norms)
    
    columns = ["h",]
    
    for i in range(fieldcount):
    
        columns += ["error{}".format(i), "order{}".format(i)]
        
    table = pandas.DataFrame(
        index = range(len(meshes)),
        columns = columns)
    
    for im, mesh in enumerate(meshes):
        
        table["h"][im] = mesh.cell_sizes((0.,)*mesh.geometric_dimension())
    
    print()
    
    print(table)
    
    
    for im, mesh in enumerate(meshes):
        
        sim = MMSVerificationSimulation(mesh = mesh, **sim_kwargs)
        
        wh = sim.solution
        
        assert(len(wh.split()) == fieldcount)
        
        if expected_orders:
    
            assert(len(expected_orders) == fieldcount)
            
        if time_dependent:
            
            sim.states = sim.run(endtime = endtime)
            
        else:
        
            sim.solution = sim.solve()
        
        w = manufactured_solution(sim)
        
        if type(w) is not tuple:
        
            w = (w,)
        
        for iw, w_i, wh_i, norm in zip(
                range(fieldcount), w, wh.split(), norms):
            
            if norm is not None:
            
                table["error{}".format(iw)][im] = fe.errornorm(
                    w_i, wh_i, norm_type = norm)
        
        if im > 0:
        
            h = table["h"]
            
            r = h[im - 1]/h[im]
            
            log = math.log
            
            for iw in range(fieldcount):
            
                e = table["error{}".format(iw)]
                
                table["order{}".format(iw)][im] = \
                    log(e[im - 1]/e[im])/log(r)
                    
        print()
        
        print(table)
        
    if outfile:
        
        print("Writing convergence table to {}".format(outfile.name))
        
        outfile.write(table.to_csv())
    
    if expected_orders:
        
        for iorder, expected_order in enumerate(expected_orders):
            
            if expected_order is None:
                
                continue
            
            order = table.iloc[-1]["order{}".format(iorder)]
            
            order = round(order, decimal_places)
            
            expected_order = round(float(expected_order), decimal_places)
            
            if not (order == expected_order):
            
                raise ValueError("\n" +
                    "\tObserved order {} differs from\n".format(order) + 
                    "\texpected order {}".format(expected_order))
                        
                    
def verify_temporal_order_of_accuracy(
        Simulation,
        manufactured_solution,
        timestep_sizes,
        endtime,
        norms,
        expected_orders = None,
        decimal_places = 2,
        sim_kwargs = {},
        starttime = 0.,
        strong_residual = None,
        dirichlet_boundary_conditions = None,
        outfile = None,
        write_simulation_outputs = False):
    
    MMSVerificationSimulation = make_mms_verification_sim_class(
        Simulation = Simulation,
        manufactured_solution = manufactured_solution,
        write_simulation_outputs = write_simulation_outputs,
        strong_residual = strong_residual,
        mms_dirichlet_boundary_conditions = dirichlet_boundary_conditions)
    
    table = sapphire.output.Table(("Delta_t", "errors", "temporal_orders"))
    
    print("")
    
    for timestep_size in timestep_sizes:
        
        sim = MMSVerificationSimulation(**sim_kwargs)
    
        assert(len(sim.solutions) > 1)
        
        sim.timestep_size = sim.timestep_size.assign(timestep_size)
        
        sim.time = sim.time.assign(starttime)
        
        sim.states = sim.run(endtime = endtime)
        
        errors = []
        
        w = manufactured_solution(sim)
        
        wh = sim.solution
        
        if type(w) is not type((0,)):
        
            w = (w,)
            
        for w_i, wh_i, norm in zip(w, wh.split(), norms):
            
            if norm is not None:
            
                errors.append(fe.errornorm(w_i, wh_i, norm_type = norm))
                
            else:
                
                errors.append(None)
                
        table.append({"Delta_t": timestep_size, "errors": errors})
        
        if len(table) > 1:
        
            Delta_t, e = table.data["Delta_t"], table.data["errors"]

            log = math.log
            
            orders = []
            
            for i in range(len(sim.solution.split())):
            
                if e[0][i] is None:
                
                    orders.append(None)
                    
                else:
            
                    r = Delta_t[-2]/Delta_t[-1]
            
                    orders.append(log(e[-2][i]/e[-1][i])/log(r))
                    
            table.data["temporal_orders"][-1] = orders
        
        print(str(table))
        
    print("Last observed temporal orders of accuracy are {}".format(orders))
    
    if outfile:
        
        print("Writing convergence table to {}".format(outfile.name))
        
        outfile.write(str(table))
        
    if expected_orders:
        
        for order, expected_order in zip(orders, expected_orders):
            
            if expected_order is None:
            
                continue
            
            order = round(order, decimal_places)
            
            expected_order = round(float(expected_order), decimal_places)
            
            if not (order == expected_order):
            
                raise ValueError("\n" +
                    "\tObserved order {} differs from\n".format(order) + 
                    "\texpected order {}".format(expected_order))
                    