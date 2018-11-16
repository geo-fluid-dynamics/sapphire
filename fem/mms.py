""" Verify via the Method of Manufactured Solution (MMS) """
import firedrake as fe
import fem.table
import math


TIME_EPSILON = 1.e-8
    
def verify_order_of_accuracy(
        Model,
        expected_spatial_order,
        grid_sizes = (8, 16, 32),
        expected_temporal_order = None,
        endtime = 1.,
        timestep_sizes = (1., 1./2., 1./4.),
        quadrature_degree = None,
        tolerance = 0.1):
    
    if expected_temporal_order is None:
    
        time_accurate = False
        
        timestep_size = None
        
    else:
    
        time_accurate = True
    
    
    class MMSVerificationModel(Model):
        
        def weak_form_residual(self):
        
            r = super().weak_form_residual()
    
            u = self.manufactured_solution()
            
            try:
            
                for psi, s_i in zip(
                        fe.TestFunctions(self.function_space),
                        self.strong_form_residual()):

                    r -= fe.inner(psi, s_i)
                    
            except NotImplementedError as error:
            
                psi = fe.TestFunction(self.function_space)
                
                s = self.strong_form_residual()
                
                r -= fe.inner(psi, s)
                
            return r
            
        def dirichlet_boundary_conditions(self):
            
            u_m = self.manufactured_solution()
            
            if time_accurate:
            
                u_m = fe.Expression(u_m)
            
                u_m.t = self.time
            
            V = self.function_space
            
            try:
    
                iterable = iter(u_m)
                
                bcs = [fe.DirichletBC(V.sub(i), g, "on_boundary") \
                    for i, g in enumerate(u_m)]
                
            except NotImplementedError as error:
                
                bcs = [fe.DirichletBC(V, u_m, "on_boundary"),]
                
            return bcs
            
        def L2_error(self):
            
            dx = self.integration_measure
            
            try:
            
                e = 0.
            
                for u_m, u_h in zip(
                        self.manufactured_solution(), self.solution.split()):
                        
                    if time_accurate:
                        
                        u_m = fe.Expression(u_m)
                        
                        u_m.t = self.time
                
                    e += fe.assemble(fe.inner(u_m - u_h, u_m - u_h)*dx)

                e = math.sqrt(e)
                
            except NotImplementedError as error:
            
                u_m, u_h = self.manufactured_solution(), self.solution
                
                if time_accurate:
                    
                    u_m = fe.Expression(u_m)
                    
                    u_m.t = self.time
                
                e = math.sqrt(fe.assemble(fe.inner(u_m - u_h, u_m - u_h)*dx))
                
            return e
    
    
    table = fem.table.Table(("h", "Delta_t", "L2_error"))
    
    
    print("Verifying spatial order of accuracy. ")
    
    for gridsize in grid_sizes:
        
        model = MMSVerificationModel(gridsize = gridsize)
        
        if time_accurate:
        
            model.time = 0.
        
            model.set_initial_values(model.manufactured_solution())
            
            model.timestep_size.assign(max(timestep_sizes))
        
            model.time += model.timestep_size.__float__()
        
        model.solve()
        
        table.append({
            "h": 1./float(model.gridsize),
            "Delta_t": timestep_size,
            "L2_error": model.L2_error()})
        
    print(str(table))
        
    e, h = table.data["L2_error"], table.data["h"]

    log = math.log

    spatial_order = log(e[-2]/e[-1])/log(h[-2]/h[-1])
    
    print("Observed spatial order of accuracy is " + str(spatial_order))
    
    assert(abs(spatial_order - expected_spatial_order) < tolerance)
    
    
    if not time_accurate:
    
        return
        
        
    print("Verifying temporal order of accuracy. ")
    
    gridsize = max(grid_sizes)
    
    for Delta_t in timestep_sizes:
        
        model = MMSVerificationModel(gridsize = grid_size)
        
        model.timestep_size.assign(Delta_t)
        
        model.time = 0.
        
        while model.time < (endtime - TIME_EPSILON):
        
            if model.time < TIME_EPSILON :
            
                model.set_initial_values(model.manufactured_solution())
            
            else:
            
                model.set_initial_values(model.solution)
        
            model.time += model.timestep_size.__float__()
            
            model.solve()
            
        table.append({
            "h": 1./float(model.gridsize), 
            "Delta_t": timestep_size,
            "L2_error": model.L2_error()})
    
    print(str(table))
    
    e, h = table.data["L2_error"], table.data["h"]
    
    temporal_order = log(e[-2]/e[-1])/log(h[-2]/h[-1])
    
    print("Observed temporal order of accuracy is " + str(temporal_order))
    
    assert(abs(temporal_order - expected_temporal_order) < tolerance)
    