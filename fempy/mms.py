""" Verify via the Method of Manufactured Solution (MMS) """
import firedrake as fe
import fempy.table
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
            
            if len(V) == 1:
            
                s = (s,)
            
            for psi, s_i in zip(fe.TestFunctions(V), s):
                
                self.weak_form_residual -= fe.inner(psi, s_i)
                
        def init_dirichlet_boundary_conditions(self):
            
            u_m = self.manufactured_solution
            
            W = self.function_space
            
            if len(W) == 1:
            
                u_m = (u_m,)
            
            self.dirichlet_boundary_conditions = [
                fe.DirichletBC(V, g, "on_boundary") 
                for V, g in zip(W, u_m)]
                
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
        parameters = {},
        timestep_size = None,
        endtime = None,
        starttime = 0.):
    
    MMSVerificationModel = make_mms_verification_model_class(Model)
    
    table = fempy.table.Table(("h", "L2_error", "spatial_order"))
    
    print("")
    
    for meshsize in grid_sizes:
        
        model = MMSVerificationModel(meshsize = meshsize)
        
        model.assign_parameters(parameters)
        
        if hasattr(model, "time"):
        
            model.time.assign(starttime)
            
            model.timestep_size.assign(timestep_size)
            
            model.run(endtime = endtime)
        
        else:
        
            model.solve()
        
        table.append({
            "h": 1./float(model.meshsize),
            "L2_error": model.L2_error()})
            
        if len(table) > 1:
        
            h, e = table.data["h"], table.data["L2_error"]

            log = math.log
            
            r = h[-2]/h[-1]
            
            order = log(e[-2]/e[-1])/log(r)
            
            table.data["spatial_order"][-1] = order
        
        print(str(table))
    
    print("Last observed spatial order of accuracy is " + str(order))
    
    assert(abs(order - expected_order) < tolerance)
    
    
def verify_temporal_order_of_accuracy(
        Model,
        expected_order,
        meshsize,
        timestep_sizes,
        endtime,
        tolerance,
        parameters = {},
        starttime = 0.):
    
    MMSVerificationModel = make_mms_verification_model_class(Model)
    
    table = fempy.table.Table(("Delta_t", "L2_error", "temporal_order"))
    
    model = MMSVerificationModel(meshsize = meshsize)
    
    model.assign_parameters(parameters)
    
    initial_values = [fe.Function(iv) for iv in model.initial_values]
    
    print("")
    
    for timestep_size in timestep_sizes:
        
        model.timestep_size.assign(timestep_size)
        
        model.time.assign(starttime)
        
        for i, iv in enumerate(model.initial_values):
        
            iv.assign(initial_values[i])
        
        model.run(endtime = endtime)
            
        table.append({
            "Delta_t": timestep_size,
            "L2_error": model.L2_error()})
            
        if len(table) > 1:
        
            Delta_t, e = table.data["Delta_t"], table.data["L2_error"]

            log = math.log
            
            r = Delta_t[-2]/Delta_t[-1]

            order = log(e[-2]/e[-1])/log(r)
    
            table.data["temporal_order"][-1] = order
        
        print(str(table))
        
    print("Last observed temporal order of accuracy is " + str(order))
    
    assert(abs(order - expected_order) < tolerance)
    