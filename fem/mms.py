""" Verify via the Method of Manufactured Solution (MMS) """
import firedrake as fe
import fem.table
import math


def verify_spatial_order_of_accuracy(
        Model,
        expected_order,
        grid_sizes = (8, 16, 32),
        quadrature_degree = None,
        tolerance = 0.1):
    
    class MMSVerificationModel(Model):
        
        def weak_form_residual(self):
        
            r = super().weak_form_residual()
            
            s = self.strong_form_residual(self.manufactured_solution())
            
            V = self.function_space
            
            if hasattr(self, "time"):
                
                s = fe.interpolate(fe.Expression(s, t = self.time), V)
            
            try:
            
                for psi, s_i in zip(fe.TestFunctions(V), s):
                    
                    r -= fe.inner(psi, s_i)
                    
            except NotImplementedError as error:
            
                psi = fe.TestFunction(V)
                
                r -= fe.inner(psi, s)
                
            return r
            
        def dirichlet_boundary_conditions(self):
            
            u_m = self.manufactured_solution()
            
            V = self.function_space
            
            if hasattr(self, "time"):
            
                u_m = fe.interpolate(fe.Expression(u_m, t = self.time), V)
            
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
                    
                    e += fe.assemble(fe.inner(u_m - u_h, u_m - u_h)*dx)

                e = math.sqrt(e)
                
            except NotImplementedError as error:
            
                u_m, u_h = self.manufactured_solution(), self.solution
                
                e = math.sqrt(fe.assemble(fe.inner(u_m - u_h, u_m - u_h)*dx))
                
            return e
    
    
    table = fem.table.Table(("h", "Delta_t", "L2_error"))
    
    
    print("Verifying spatial order of accuracy.")
    
    for gridsize in grid_sizes:
        
        model = MMSVerificationModel(gridsize = gridsize)
        
        model.solve()
        
        table.append({
            "h": 1./float(model.gridsize),
            "L2_error": model.L2_error()})
        
    print(str(table))
        
    e, h = table.data["L2_error"], table.data["h"]

    log = math.log

    spatial_order = log(e[-2]/e[-1])/log(h[-2]/h[-1])
    
    print("Observed spatial order of accuracy is " + str(spatial_order))
    
    assert(abs(spatial_order - expected_order) < tolerance)
    