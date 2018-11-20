""" Verify via the Method of Manufactured Solution (MMS) """
import firedrake as fe
import fem.table
import math


TIME_EPSILON = 1.e-8

def make_mms_verification_model_class(Model):

    class MMSVerificationModel(Model):
        
        def init_mesh(self):
        
            super().init_mesh()
            
            self.init_manufactured_solution()
            
        def init_weak_form_residual(self):
        
            super().init_weak_form_residual()
            
            s = self.strong_form_residual(self.manufactured_solution)
            
            V = self.function_space
            
            try:
            
                for psi, s_i in zip(fe.TestFunctions(V), s):
                    
                    self.weak_form_residual -= fe.inner(psi, s_i)
                    
            except NotImplementedError as error:
            
                psi = fe.TestFunction(V)
                
                self.weak_form_residual -= fe.inner(psi, s)
            
        def init_dirichlet_boundary_conditions(self):
            
            u_m = self.manufactured_solution
            
            V = self.function_space
            
            try:
    
                iterable = iter(u_m)
                
                bcs = [fe.DirichletBC(V.sub(i), g, "on_boundary") \
                    for i, g in enumerate(u_m)]
                
            except NotImplementedError as error:
                
                bcs = [fe.DirichletBC(V, u_m, "on_boundary"),]
                
            self.dirichlet_boundary_conditions = bcs
            
        def L2_error(self):
            
            dx = self.integration_measure
            
            try:
            
                e = 0.
            
                for u_h, u_m in zip(
                        self.solution.split(), self.manufactured_solution):
                    
                    e += fe.assemble(fe.inner(u_h - u_m, u_h - u_m)*dx)

                e = math.sqrt(e)
                
            except NotImplementedError as error:
            
                u_h, u_m = self.solution, self.manufactured_solution
                
                e = math.sqrt(fe.assemble(fe.inner(u_h - u_m, u_h - u_m)*dx))
                
            return e
            
    return MMSVerificationModel
    
    
def verify_spatial_order_of_accuracy(
        Model,
        expected_order,
        grid_sizes,
        tolerance,
        timestep_size = None):
    
    MMSVerificationModel = make_mms_verification_model_class(Model)
    
    table = fem.table.Table(("h", "L2_error", "spatial_order"))
    
    for gridsize in grid_sizes:
        
        model = MMSVerificationModel(gridsize = gridsize)
        
        if hasattr(model, "time"):
            
            model.initial_values.assign(fe.interpolate(
                model.manufactured_solution, model.function_space))
            
            model.timestep_size.assign(timestep_size)
            
            model.time.assign(model.time + model.timestep_size)
        
        model.solver.solve()
        
        table.append({
            "h": 1./float(model.gridsize),
            "L2_error": model.L2_error()})
            
        if len(table) > 1:
        
            h, e = table.data["h"], table.data["L2_error"]

            log = math.log

            order = log(e[-2]/e[-1])/log(h[-2]/h[-1])
            
            table.data["spatial_order"][-1] = order
        
    print(str(table))
    
    max_order = table.max("spatial_order")
    
    print("Maximum observed spatial order of accuracy is " + str(max_order))
    
    assert(abs(max_order - expected_order) < tolerance)
    
    
def verify_temporal_order_of_accuracy(
        Model,
        expected_order,
        gridsize,
        timestep_sizes,
        endtime,
        tolerance):
    
    MMSVerificationModel = make_mms_verification_model_class(Model)
    
    table = fem.table.Table(("Delta_t", "L2_error", "temporal_order"))
    
    model = MMSVerificationModel(gridsize = gridsize)
    
    initial_values = fe.interpolate(
        model.manufactured_solution, model.function_space)
    
    initial_time = model.time.__float__()
    
    for timestep_size in timestep_sizes:
        
        model.timestep_size.assign(timestep_size)
        
        time = initial_time
        
        model.time.assign(time)
        
        model.initial_values.assign(initial_values)
        
        while time < (endtime - TIME_EPSILON):
            
            time += timestep_size
            
            model.time.assign(time)
            
            model.solver.solve()
            
            model.initial_values.assign(model.solution)
            
        table.append({
            "Delta_t": timestep_size,
            "L2_error": model.L2_error()})
            
        if len(table) > 1:
        
            Delta_t, e = table.data["Delta_t"], table.data["L2_error"]

            log = math.log

            order = log(e[-2]/e[-1])/log(Delta_t[-2]/Delta_t[-1])
    
            table.data["temporal_order"][-1] = order
        
    print(str(table))
    
    max_order = table.max("temporal_order")
    
    print("Maximum observed temporal order of accuracy is " + str(max_order))
    
    assert(abs(max_order - expected_order) < tolerance)
    