import firedrake as fe
import fempy.model


def solve(
        model,
        solver,
        continuation_parameter,
        continuation_sequence,
        leftval,
        rightval,
        startleft = False,
        maxcount = 32):
    """ Solve a strongly nonlinear problem 
    by solving a sequence of over-regularized problems 
    with successively reduced regularization.
    Here we use the word 'smooth' as a short synonym for 'regularize'.
    
    Always continue from left to right.
    """
    if continuation_sequence is None:
    
        my_continuation_sequence = (leftval, rightval)
        
    else:
    
        my_continuation_sequence = continuation_sequence
    
    if startleft:
    
        first_s_to_solve = my_continuation_sequence[0]
    
    else:
    
        first_s_to_solve = my_continuation_sequence[-1]
        
    attempts = range(maxcount - len(my_continuation_sequence))
    
    solved = False
    
    def bounded(val):
    
        if leftval < rightval:
        
            return leftval <= val and val <= rightval
            
        if leftval > rightval:
        
            return leftval >= val and val >= rightval
            
    for attempt in attempts:

        s_start_index = my_continuation_sequence.index(first_s_to_solve)
        
        try:
        
            for s in my_continuation_sequence[s_start_index:]:
                
                continuation_parameter.assign(s)
                
                model.backup_solution.assign(model.solution)
                
                solver.solve()
                
                print("Solved with continuation parameter = " + str(s))
                
            solved = True
            
            break
            
        except fe.exceptions.ConvergenceError:  
            
            current_s = continuation_parameter.__float__()
            
            ss = my_continuation_sequence
        
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
            
            model.solution.assign(model.backup_solution)
            
            my_continuation_sequence = new_ss
            
            print("Inserted new value of " + str(s_to_insert))
            
            first_s_to_solve = s_to_insert
    
    assert(solved)
    
    assert(continuation_parameter.__float__() ==
        my_continuation_sequence[-1])
    
    return my_continuation_sequence
    