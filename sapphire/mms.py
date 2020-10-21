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
import numpy
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
    

def format_for_latex(table, norms):
    
    formatted_table = table.copy()
    
    for i, norm in enumerate(norms):
        
        for label, format in zip(
                ('error', 'order'),
                ('{:.3e}', '{:.3f}')):
            
            column = label + str(i)
            
            if norm is not None:
            
                formatted_table[column] = pandas.Series(
                    [format.format(val) 
                     for val in table[column]],
                    index = table.index)
    
    return formatted_table.to_latex(index=False).replace("nan", "   ")
    

def verify_order_of_accuracy(
        discretization_parameter_name,
        discretization_parameter_values,
        Simulation,
        manufactured_solution,
        norms,
        points_in_rate_estimator = 2,
        expected_orders = None,  # a.k.a. convergence rates
        decimal_places = 2,
        time_dependent = True,
        sim_kwargs = {},
        endtime = 0.,
        strong_residual = None,
        dirichlet_boundary_conditions = None,
        starttime = 0.,
        outfile = None,
        write_simulation_outputs = False):
    
    n = points_in_rate_estimator
    
    pname = discretization_parameter_name
    
    pvalues = discretization_parameter_values
    
    
    if n > 3:
    
        raise NotImplementedError("Max `points_in_rate_estimator` is 3.")
    
    if len(pvalues) < n:
    
        raise ValueError("There must be a discretization parameter value "
                         "for each point in the rate estimation.")
    
    
    fieldcount = len(norms)
    
    if expected_orders:
        
        assert(len(expected_orders) == len(norms))
        
    
    for pvalue in pvalues:
    
        if not pvalue > 0:
        
            raise("`discretization_parameter_values` must be positive.")
            
    
    r = pvalues[0]/pvalues[1]  # Refinement rate
    
    if r < 0:
    
        raise("`discretization_parameter_values` must be "
              "a descending sequence.")
    
    if n > 2:
        
        for iv in range(1, len(pvalues)):
            
            if not numpy.isclose(pvalues[iv - 1]/pvalues[iv], r):

                raise("`discretization_parameter_values` must be "
                      "a geometric sequence when using more than two "
                      "points in the rate estimator.")
        
    
    MMSVerificationSimulation = make_mms_verification_sim_class(
        Simulation = Simulation,
        manufactured_solution = manufactured_solution,
        write_simulation_outputs = write_simulation_outputs,
        strong_residual = strong_residual,
        mms_dirichlet_boundary_conditions = dirichlet_boundary_conditions)
    
    
    columns = [pname,]
    
    for i, norm in enumerate(norms):
    
        if norm is not None:
        
            columns += ["error{}".format(i), "order{}".format(i)]
    
    table = pandas.DataFrame(
        index = range(len(pvalues)),
        columns = columns)
    
    for iv, pval in enumerate(pvalues):
        
        table[pname][iv] = pval
    
    print()
    
    print(str(table).replace(" NaN", "None"))
    
    
    for iv, pval in enumerate(pvalues):
        
        sim_kwargs[pname] = pval
        
        sim = MMSVerificationSimulation(**sim_kwargs)
        
        wh = sim.solution
        
        assert(len(wh.split()) == fieldcount)
            
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
            
                table["error{}".format(iw)][iv] = fe.errornorm(
                    w_i, wh_i, norm_type = norm)
        
        if iv >= (n - 1):
            
            h = table[pname]
            
            log = math.log
            
            for iw in range(fieldcount):
                
                if norms[iw] is not None:
                    
                    e = table["error{}".format(iw)]
                    
                    if n == 2:
                        
                        order = log(e[iv - 1]/e[iv])/log(r)
                        
                    elif n == 3:
                    
                        order = log((e[iv - 2] - e[iv - 1]) /
                            (e[iv - 1] - e[iv]))/log(r)
                        
                    table["order{}".format(iw)][iv] = order
            
        print()
        
        print(str(table).replace(" NaN", "None"))
        
    if outfile:
        
        print("Writing convergence table to {}".format(outfile.name))
        
        outfile.write(table.to_csv())
    
    if expected_orders:
        
        for iorder, expected_order in enumerate(expected_orders):
            
            if expected_order is not None:
            
                order = table.iloc[-1]["order{}".format(iorder)]
                
                order = round(order, decimal_places)
                
                expected_order = round(float(expected_order), decimal_places)
                
                if not (order == expected_order):
                
                    raise ValueError("\n" +
                        "\tObserved order {} differs from\n".format(order) + 
                        "\texpected order {}".format(expected_order))

    print()
    
    print("Formatted for LaTeX:")
    
    print(format_for_latex(table, norms=norms))
    
    return table
    