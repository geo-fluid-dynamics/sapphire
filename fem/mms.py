import firedrake as fe
import math


def augment_weak_form(
        weak_form_residual, 
        function_space, 
        strong_form_residual, 
        manufactured_solution):
    
    R = weak_form_residual()
    
    for psi, s in zip(
            fe.TestFunctions(function_space), 
            strong_form_residual(
                manufactured_solution(function_space.mesh()))):

        R -= fe.inner(psi, s)*fe.dx
        
    return R

def L2_error(manufactured_solution, computed_solution):
    
    L2_error = 0.
    
    for u_m, u_h in zip(manufactured_solution, computed_solution.split()):
    
        L2_error += fe.assemble(fe.inner(u_m - u_h, u_m - u_h)*fe.dx)
    
    return math.sqrt(L2_error)
    
    
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
        
        L2_errors.append(
            L2_error(manufactured_solution(mesh), model.solution))
    
    edge_lengths = [1./float(M) for M in grid_sizes]

    e, h = L2_errors, edge_lengths

    log = math.log

    orders = [(log(e[i + 1]) - log(e[i]))/(log(h[i + 1]) - log(h[i]))
              for i in range(len(e) - 1)]
    
    print("Edge lengths = " + str(edge_lengths))
    
    print("L2 norm errors = " + str(L2_errors))
    
    print("Convergence orders = " + str(orders))
    
    assert(abs(orders[-1] - expected_order) < tolerance)
    