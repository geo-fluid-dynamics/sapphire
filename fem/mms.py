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
    
    print("")
    
    for gridsize in grid_sizes:
        
        model = MMSVerificationModel(gridsize = gridsize)
        
        if hasattr(model, "time"):
            
            initial_values = fe.interpolate(
                model.manufactured_solution, model.function_space)
                
            for iv in model.initial_values:
            
                iv.assign(initial_values)
            
            model.timestep_size.assign(timestep_size)
            
            model.time.assign(model.time + model.timestep_size)
        
        model.solver.solve()
        
        table.append({
            "h": 1./float(model.gridsize),
            "L2_error": model.L2_error()})
            
        if len(table) > 1:
        
            h, e = table.data["h"], table.data["L2_error"]

            log = math.log
            
            r = h[-2]/h[-1]
            
            order = log(e[-2]/e[-1])/log(r)
            
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
        tolerance,
        starttime = 0.,
        plot_solutions = False):
    
    if plot_solutions:
    
        import matplotlib
        
        import matplotlib.pyplot as pp
        
    MMSVerificationModel = make_mms_verification_model_class(Model)
    
    table = fem.table.Table(("Delta_t", "L2_error", "temporal_order"))
    
    model = MMSVerificationModel(gridsize = gridsize)
    
    model.time.assign(starttime)
    
    initial_values = fe.interpolate(
        model.manufactured_solution, model.function_space)
    
    initial_time = model.time.__float__()
    
    print("")
    
    for timestep_size in timestep_sizes:
        
        model.timestep_size.assign(timestep_size)
        
        time = starttime
        
        model.time.assign(time)
        
        for iv in model.initial_values:
        
            iv.assign(initial_values)
        
        while time < (endtime - TIME_EPSILON):
            
            time += timestep_size
            
            model.time.assign(time)
            
            model.solver.solve()
            
            for i in range(len(model.initial_values) - 1):
        
                model.initial_values[-i - 1].assign(
                    model.initial_values[-i - 2])
                    
            model.initial_values[0].assign(model.solution)
            
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
        
        if plot_solutions:
        
            assert(model.mesh.geometric_dimension() == 1)
        
            figure = pp.figure()
            
            axes = pp.axes()
            
            fe.plot(model.solution, axes = axes, color = "red")
            
            u_m = fe.interpolate(
                model.manufactured_solution, model.function_space)
            
            line = fe.plot(u_m, axes = axes, color = "blue")
            
            pp.axis("square")
            
            pp.legend((r"$u_h$", r"$u_m$"))
            
            pp.xlabel(r"$x$")
            
            pp.ylabel(r"$u$")
            
            pp.savefig("uh_vs_um__Delta_t_" + str(timestep_size) + ".png")
    
    max_order = table.max("temporal_order")
    
    print("Maximum observed temporal order of accuracy is " + str(max_order))
    
    assert(abs(max_order - expected_order) < tolerance)
    