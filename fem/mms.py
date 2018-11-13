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

            r -= fe.inner(psi, s_i)*fe.dx
            
    except NotImplementedError as error:
    
        psi = fe.TestFunction(function_space)
        
        s = strong_form_residual(manufactured_solution(function_space.mesh()))
                    
        r -= fe.inner(psi, s)*fe.dx
        
    return r

def L2_error(manufactured_solution, computed_solution):
    
    mesh = computed_solution.function_space().mesh()
    
    try:
    
        L2_error = 0.
    
        for u_m, u_h in zip(manufactured_solution(mesh), computed_solution.split()):
        
            L2_error += fe.assemble(fe.inner(u_m - u_h, u_m - u_h)*fe.dx)

        L2_error = math.sqrt(L2_error)
        
    except NotImplementedError as error:
    
        u_m, u_h = manufactured_solution(mesh), computed_solution
        
        L2_error = math.sqrt(fe.assemble(fe.inner(u_m - u_h, u_m - u_h)*fe.dx))
        
    return L2_error
    
    
def verify_convergence_order(
        Model,
        expected_order,
        strong_form_residual,
        manufactured_solution,
        grid_sizes = (8, 16, 32), 
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
        
        model = MMSVerificationModel(
            mesh = mesh,
            boundary_condition_values = manufactured_solution(mesh))
        
        model.solver.solve()
        
        L2_errors.append(L2_error(manufactured_solution, model.solution))
    
    edge_lengths = [1./float(M) for M in grid_sizes]

    e, h = L2_errors, edge_lengths

    log = math.log

    orders = [(log(e[i + 1]) - log(e[i]))/(log(h[i + 1]) - log(h[i]))
              for i in range(len(e) - 1)]
    
    print("Edge lengths = " + str(edge_lengths))
    
    print("L2 norm errors = " + str(L2_errors))
    
    print("Convergence orders = " + str(orders))
    
    assert(abs(orders[-1] - expected_order) < tolerance)
    