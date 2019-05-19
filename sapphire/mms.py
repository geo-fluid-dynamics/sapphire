""" Verify a FEM simulation via the Method of Manufactured Solution (MMS).

This module assumes that the FEM simulation solves a variational problem
which approximates solutions to a strong form PDE.
"""
import firedrake as fe
import sapphire.table
import math
import pathlib


def mms_source(
        sim,
        strong_residual,
        manufactured_solution):
    
    V = sim.solution.function_space()
    
    _r = strong_residual(sim = sim, solution = manufactured_solution(sim))
        
    if type(sim.element) is fe.FiniteElement:
    
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

    initial_values = fe.Function(sim.function_space)
    
    if type(sim.element) is fe.FiniteElement:
    
        w_m = (manufactured_solution,)
        
    else:
    
        w_m = manufactured_solution
        
    for u_m, V in zip(
            w_m, sim.function_space):
        
        initial_values.assign(fe.interpolate(u_m, V))
        
    return initial_values
    
    
def mms_dirichlet_boundary_conditions(sim, manufactured_solution):
    
    W = sim.function_space
    
    if type(sim.element) is fe.FiniteElement:
    
        w = (manufactured_solution,)
    
    else:
    
        w = manufactured_solution
        
    return [fe.DirichletBC(V, g, "on_boundary") for V, g in zip(W, w)]
    
    
def make_mms_verification_sim_class(
        sim_module,
        manufactured_solution):
    
    def initial_values(sim):
        
        return mms_initial_values(
            sim = sim,
            manufactured_solution = manufactured_solution(sim))
        
    def dirichlet_boundary_conditions(sim):
    
        return mms_dirichlet_boundary_conditions(
            sim = sim,
            manufactured_solution = manufactured_solution(sim))
        
    class MMSVerificationSimulation(sim_module.Simulation):
            
        def __init__(self, *args, **kwargs):
            
            super().__init__(*args, 
                initial_values = initial_values,
                dirichlet_boundary_conditions = dirichlet_boundary_conditions,
                **kwargs)
                
            self.variational_form_residual -= mms_source(
                    sim = self,
                    strong_residual = sim_module.strong_residual,
                    manufactured_solution = manufactured_solution)\
                *fe.dx(degree = self.quadrature_degree)
                
        def write_outputs(self, *args, **kwargs):
        
            pass
        
    return MMSVerificationSimulation
    
    
def errornorm(w, wh, *args, **kwargs):
    """ Extends fe.errornorm to handle mixed FEM functions """
    if len(wh.split()) == 1:
    
        return fe.errornorm(w, wh.split()[0], *args, **kwargs)
    
    else:
        
        e = 0.
        
        for w_i, wh_i in zip(w, wh.split()):
            
            e += fe.errornorm(w_i, wh_i, *args, **kwargs)
        
        return e
    
    
def verify_spatial_order_of_accuracy(
        sim_module,
        manufactured_solution,
        meshes,
        expected_order,
        tolerance,
        sim_constructor_kwargs = {},
        parameters = {},
        timestep_size = 1.e32,
        endtime = 0.,
        starttime = 0.):
    
    MMSVerificationSimulation = make_mms_verification_sim_class(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution)
    
    table = sapphire.table.Table(("h", "L2_error", "spatial_order"))
    
    print("")
    
    for mesh in meshes:
        
        h = mesh.cell_sizes((0.,)*mesh.geometric_dimension())
        
        sim = MMSVerificationSimulation(mesh = mesh, **sim_constructor_kwargs)
        
        sim = sim.assign_parameters(parameters)
        
        if sim.time is not None:
            
            sim.time = sim.time.assign(starttime)
            
            sim.timestep_size = sim.timestep_size.assign(timestep_size)
            
            sim.solutions, _ = sim.run(endtime = endtime)
            
        else:
        
            sim.solution = sim.solve()
            
        table.append({
            "h": h,
            "L2_error": 
                errornorm(
                    manufactured_solution(sim),
                    sim.solution,
                    norm_type = "L2")})
            
        if len(table) > 1:
        
            h, e = table.data["h"], table.data["L2_error"]

            log = math.log
            
            r = h[-2]/h[-1]
            
            order = log(e[-2]/e[-1])/log(r)
            
            table.data["spatial_order"][-1] = order
        
        print(str(table))
    
    print("Last observed spatial order of accuracy is {0}".format(order))
    
    assert(abs(order - expected_order) < tolerance)
    
    
def verify_temporal_order_of_accuracy(
        sim_module,
        manufactured_solution,
        mesh,
        timestep_sizes,
        endtime,
        expected_order,
        tolerance,
        sim_constructor_kwargs = {},
        parameters = {},
        starttime = 0.):
    
    MMSVerificationSimulation = make_mms_verification_sim_class(
        sim_module = sim_module,
        manufactured_solution = manufactured_solution)
    
    table = sapphire.table.Table(("Delta_t", "L2_error", "temporal_order"))
    
    print("")
    
    for timestep_size in timestep_sizes:
        
        sim = MMSVerificationSimulation(mesh = mesh, **sim_constructor_kwargs)
    
        assert(len(sim.solutions) > 1)
        
        sim = sim.assign_parameters(parameters)
    
        sim.timestep_size = sim.timestep_size.assign(timestep_size)
        
        sim.time = sim.time.assign(starttime)
        
        for solution in sim.solutions:
            
            solution = solution.assign(sim.initial_values)
            
        sim.solutions, _, = sim.run(endtime = endtime)
        
        table.append({
            "Delta_t": timestep_size,
            "L2_error": errornorm(
                    manufactured_solution(sim),
                    sim.solution,
                    norm_type = "L2")})
            
        if len(table) > 1:
        
            Delta_t, e = table.data["Delta_t"], table.data["L2_error"]

            log = math.log
            
            r = Delta_t[-2]/Delta_t[-1]

            order = log(e[-2]/e[-1])/log(r)
    
            table.data["temporal_order"][-1] = order
        
        print(str(table))
        
    print("Last observed temporal order of accuracy is {0}".format(order))
    
    assert(abs(order - expected_order) < tolerance)
    