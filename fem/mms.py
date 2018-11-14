""" **mms.py**
facilitates solver verification via the Method of Manufactured Solution (MMS).
"""
import firedrake as fe
import fem.table
import math


TIME_EPSILON = 1.e-8

def augment_weak_form(
        weak_form_residual, 
        function_space, 
        strong_form_residual, 
        manufactured_solution):
    
    r = weak_form_residual()
    
    mesh = function_space.mesh()
    
    u = manufactured_solution(mesh)
    
    try:
    
        for psi, s_i in zip(
                fe.TestFunctions(function_space), 
                strong_form_residual(u, mesh)):

            r -= fe.inner(psi, s_i)
            
    except NotImplementedError as error:
    
        psi = fe.TestFunction(function_space)
        
        s = strong_form_residual(u, mesh)
        
        r -= fe.inner(psi, s)
        
    return r

def L2_error(manufactured_solution, computed_solution, quadrature_degree = None):
    
    mesh = computed_solution.function_space().mesh()
    
    dx = fe.dx(degree = quadrature_degree)
    
    try:
    
        L2_error = 0.
    
        for u_m, u_h in zip(manufactured_solution(mesh), computed_solution.split()):
        
            L2_error += fe.assemble(fe.inner(u_m - u_h, u_m - u_h)*dx)

        L2_error = math.sqrt(L2_error)
        
    except NotImplementedError as error:
    
        u_m, u_h = manufactured_solution(mesh), computed_solution
        
        L2_error = math.sqrt(fe.assemble(fe.inner(u_m - u_h, u_m - u_h)*dx))
        
    return L2_error
    
    
def boundary_conditions(mesh, u_m):
        
    try:
    
        iterable = iter(u_m)
    
        bcs = [{"subspace": i, "value": g, "subdomain": "on_boundary"}
            for i, g in enumerate(u_m)]
        
    except NotImplementedError as error:
    
        bcs = [{"subspace": None, "value": u_m, "subdomain": "on_boundary"},]
        
    return bcs

    
def verify_orders_of_accuracy(
        Model,
        expected_spatial_order,
        expected_temporal_order,
        strong_form_residual,
        manufactured_solution,
        grid_sizes = (8, 16, 32),
        endtime = 1.,
        timestep_sizes = (1., 1./2., 1./4.),
        quadrature_degree = None,
        residual_parameters = {},
        tolerance = 0.1):
    
    class MMSVerificationModel(Model):
    
        def weak_form_residual(self):
        
            return augment_weak_form(
                super().weak_form_residual, 
                self.solution.function_space(),
                strong_form_residual, 
                manufactured_solution)
    
    table = fem.table.Table(("h", "Delta_t", "L2_error"))
    
    """ Check spatial order of accuracy. """
    timestep_size = max(timestep_sizes)
    
    time = 0.
    
    for M in grid_sizes:
        
        mesh = fe.UnitSquareMesh(M, M)
        
        u_m = fe.Expression(manufactured_solution(mesh))
        
        u_m.t = time + timestep_size
        
        model = MMSVerificationModel(
            mesh = mesh, 
            dirichlet_boundary_conditions = boundary_conditions(mesh, u_m),
            quadrature_degree = quadrature_degree,
            residual_parameters = residual_parameters)
        
        model.set_initial_values(u_m)
        
        model.timestep_size.assign(timestep_size)
        
        model.solver.solve()
        
        table.append({
            "h": 1./float(M), 
            "Delta_t": Delta_t, 
            "L2_error": L2_error(
                manufactured_solution, 
                model.solution, 
                quadrature_degree = quadrature_degree)})
        
    print(str(table))
        
    e, h = table["L2_error"], table["h"]

    log = math.log

    spatial_order = log(e[-2]/e[-1])/log(h[-2]/h[-1])
    
    """ Check temporal order of accuracy. """
    M = max(grid_sizes)
    
    for Delta_t in timestep_sizes:
    
        time = 0.
        
        mesh = fe.UnitSquareMesh(M, M)
        
        u_m = fe.Expression(manufactured_solution(mesh))
        
        u_h = None
        
        while time < (endtime - TIME_EPSILON):
        
            u_m.t = time + timestep_size
            
            model = MMSVerificationModel(
                mesh = mesh, 
                dirichlet_boundary_conditions = boundary_conditions(mesh, u_m),
                quadrature_degree = quadrature_degree,
                residual_parameters = residual_parameters)
            
            if u_h is None:
            
                model.set_initial_values(u_m)
            
            else:
            
                model.set_initial_values(u_h)
                
            model.timestep_size.assign(Delta_t)
            
            model.solver.solve()
            
            time += timestep_size
            
            u_h = fe.Function(model.solution)
        
        table.append({
            "h": 1./float(M), 
            "Delta_t": Delta_t, 
            "L2_error": L2_error(
                manufactured_solution, 
                model.solution, 
                quadrature_degree = quadrature_degree)})
    
    print(str(table))
    
    e, h = table["L2_error"], table["h"]
    
    temporal_order = log(e[-2]/e[-1])/log(h[-2]/h[-1])
    
    """ Assert expected orders of accuracy. """
    assert(abs(spatial_order - expected_spatial_order) < tolerance)
    
    assert(abs(temporal_order - expected_temporal_order) < tolerance)
    