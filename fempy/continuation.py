import firedrake as fe
import fempy.model


def solve_with_auto_continuation(
        solve,
        solution,
        continuation_parameter,
        continuation_sequence,
        startleft = False,
        maxcount = 32):
    """ Solve a strongly nonlinear problem 
    by solving a sequence of over-regularized problems 
    with successively reduced regularization.
    
    Always continue from left to right.
    """
    if startleft:
    
        first_s_to_solve = continuation_sequence[0]
    
    else:
    
        first_s_to_solve = continuation_sequence[-1]
        
    attempts = range(maxcount - len(continuation_sequence))
    
    solved = False
    
    leftval, rightval = continuation_sequence[0], continuation_sequence[-1]
    
    def bounded(val):
    
        if leftval < rightval:
        
            return leftval <= val and val <= rightval
            
        if leftval > rightval:
        
            return leftval >= val and val >= rightval
            
    backup_solution = fe.Function(solution)
    
    snes_iteration_count = 0
    
    for attempt in attempts:

        s_start_index = continuation_sequence.index(first_s_to_solve)
        
        try:
        
            for s in continuation_sequence[s_start_index:]:
                
                continuation_parameter.assign(s)
                
                backup_solution = backup_solution.assign(solution)
                
                solution, snes_iteration_count = solve()
                
                print("Solved with continuation parameter = " + str(s))
                
            solved = True
            
            break
            
        except fe.exceptions.ConvergenceError:  
            
            current_s = continuation_parameter.__float__()
            
            ss = continuation_sequence
        
            print("Failed to solve with continuation paramter = " 
                + str(current_s) +
                " from the sequence " + str(ss))
        
            if attempt == attempts[-1]:
                
                break
            
            index = ss.index(current_s)
            
            assert(index > 0)
            
            s_to_insert = (current_s + ss[index - 1])/2.
            
            assert(bounded(s_to_insert))
            
            new_ss = ss[:index] + (s_to_insert,) + ss[index:]
            
            solution = solution.assign(backup_solution)
            
            continuation_sequence = new_ss
            
            print("Inserted new value of " + str(s_to_insert))
            
            first_s_to_solve = s_to_insert
    
    assert(solved)
    
    assert(continuation_parameter.__float__() ==
        continuation_sequence[-1])
    
    return solution, snes_iteration_count, continuation_sequence
    