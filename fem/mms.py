""" **mms.py**
facilitates solver verification via the Method of Manufactured Solution (MMS).
"""
import firedrake as fe
import math


def augment_weak_form(
        weak_form_residual, 
        function_space, 
        strong_form_residual, 
        manufactured_solution):
    
    r = weak_form_residual()
    
    try:
    
        for psi, s_i in zip(
                fe.TestFunctions(function_space), 
                strong_form_residual(
                    manufactured_solution(function_space.mesh()))):

            r -= fe.inner(psi, s_i)
            
    except NotImplementedError as error:
    
        psi = fe.TestFunction(function_space)
        
        s = strong_form_residual(manufactured_solution(function_space.mesh()))
                    
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
    
    
def verify_convergence_order(
        Model,
        expected_order,
        strong_form_residual,
        manufactured_solution,
        grid_sizes = (8, 16, 32),
        quadrature_degree = None,
        tolerance = 0.1):
    
    class MMSVerificationModel(Model):
    
        def weak_form_residual(self):
        
            return augment_weak_form(
                super().weak_form_residual, 
                self.solution.function_space(),
                strong_form_residual, 
                manufactured_solution)
            
    L2_errors = []

    for M in grid_sizes:
        
        mesh = fe.UnitSquareMesh(M, M)
        
        u = manufactured_solution(mesh)
        
        try:
        
            iterable = iter(u)
        
            bcs = [{"subspace": i, "value": g, "subdomain": "on_boundary"}
                for i, g in enumerate(u)]
            
        except NotImplementedError as error:
        
            bcs = [{"subspace": None, "value": u, "subdomain": "on_boundary"},]
        
        model = MMSVerificationModel(
            mesh = mesh, 
            dirichlet_boundary_conditions = bcs,
            quadrature_degree = quadrature_degree)
        
        model.solver.solve()
        
        L2_errors.append(
            L2_error(
                manufactured_solution, 
                model.solution, 
                quadrature_degree = quadrature_degree))
    
    edge_lengths = [1./float(M) for M in grid_sizes]

    e, h = L2_errors, edge_lengths

    log = math.log

    orders = [(log(e[i + 1]) - log(e[i]))/(log(h[i + 1]) - log(h[i]))
              for i in range(len(e) - 1)]
    
    print("Edge lengths = " + str(edge_lengths))
    
    print("L2 norm errors = " + str(L2_errors))
    
    print("Convergence orders = " + str(orders))
    
    assert(abs(orders[-1] - expected_order) < tolerance)
    