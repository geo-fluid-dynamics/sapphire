"""Nonlinear solver continuation

for regularized nonlinear problems.
"""
import firedrake as fe


def solve_with_over_regularization(
        solve,
        solution,
        regularization_parameter,
        search_operator = lambda r: 2.*r,
        attempts = 24,
        startval = None):
    
    original_regularization_parameter_value = \
        regularization_parameter.__float__()
        
    if startval is None:
    
        r0 = original_regularization_parameter_value
        
    else:
    
        r0 = startval
    
    backup_solution = fe.Function(solution)
    
    print("Searching for a working regularization")
    
    r = r0
    
    for attempt in range(attempts):
        
        regularization_parameter = regularization_parameter.assign(r)
        
        print("Trying r = {}".format(r))
        
        try:
            
            solution = solve()
            
            regularization_parameter = regularization_parameter.assign(     
                original_regularization_parameter_value)
            
            return solution, r
            
        except fe.exceptions.ConvergenceError as exception:
            
            r = search_operator(r)
            
            solution = solution.assign(backup_solution)
            
            if attempt == range(attempts)[-1]:
            
                raise(exception)
                
def solve_with_bounded_regularization_sequence(
        solve,
        solution,
        regularization_parameter,
        initial_regularization_sequence,
        backup_solution = None,
        maxcount = 24):
    """ Solve a strongly nonlinear problem 
    by solving a sequence of over-regularized problems 
    with successively reduced regularization.
    
    Always continue from left to right.
    """
    if backup_solution is None:
    
        backup_solution = fe.Function(solution)
        
    r0 = regularization_parameter.__float__()
    
    assert(initial_regularization_sequence[-1] == r0)
    
    regularization_sequence = initial_regularization_sequence
    
    first_r_to_solve = regularization_sequence[0]
    
    attempts = range(maxcount - len(regularization_sequence))
    
    solved = False
    
    backup_solution = backup_solution.assign(solution)
    
    for attempt in attempts:

        r_start_index = regularization_sequence.index(first_r_to_solve)
        
        try:
        
            for r in regularization_sequence[r_start_index:]:
                
                regularization_parameter.assign(r)
                
                solution = solve()
                
                backup_solution = backup_solution.assign(solution)
                
                print("Solved with continuation parameter = {}".format(r))
                
            solved = True
            
            break
            
        except fe.exceptions.ConvergenceError as exception:
            
            current_r = regularization_parameter.__float__()
            
            rs = regularization_sequence
        
            print("Failed to solve with continuation parameter = {}"
                  " from the sequence {}".format(current_r, rs))
                
            index = rs.index(current_r)
            
            if attempt == attempts[-1] or (index == 0):
                
                regularization_parameter = regularization_parameter.assign(r0)
                
                raise(exception)
            
            solution = solution.assign(backup_solution)
            
            r_to_insert = (current_r + rs[index - 1])/2.
            
            new_rs = rs[:index] + (r_to_insert,) + rs[index:]
            
            regularization_sequence = new_rs
            
            print("Inserted new value of " + str(r_to_insert))
            
            first_r_to_solve = r_to_insert
            
    assert(solved)
    
    assert(regularization_parameter.__float__() == r0)
    
    assert(regularization_sequence[-1] == r0)
    
    return solution, regularization_sequence
    