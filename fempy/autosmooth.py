""" Solve a strongly nonlinear problem by solving a sequence of over-regularized
problems with successively reduced regularization.
Here we use the word 'smooth' as a shorter synonym for 'regularize'.
"""
import firedrake as fe
import fempy.model


def solve(
        model,
        maxval = 4.,
        firstval = None,
        maxcount = 32):
    
    if model.smoothing_sequence is None:
    
        smoothing_sequence = (model.smoothing.__float__(),)
        
        if firstval is None:
        
            firstval = model.smoothing.__float__()
        
        while smoothing_sequence[0] < firstval:

            smoothing_sequence = (2.*smoothing_sequence[0],) + \
                smoothing_sequence
        
    else:
    
        smoothing_sequence = model.smoothing_sequence
    
    first_s_to_solve = smoothing_sequence[0]
    
    attempts = range(maxcount - len(smoothing_sequence))
    
    solved = False
    
    for attempt in attempts:

        s_start_index = smoothing_sequence.index(first_s_to_solve)
        
        try:
        
            for s in smoothing_sequence[s_start_index:]:
                
                model.smoothing.assign(s)
                
                model.backup_solution.assign(model.solution)
                
                fempy.model.Model.solve(model)
                
                if not model.quiet:
                
                    print("Solved with s = " + str(s))
                
            solved = True
            
            break
            
        except fe.exceptions.ConvergenceError:  
            
            current_s = model.smoothing.__float__()
            
            ss = smoothing_sequence
            
            if not model.quiet:
            
                print("Failed to solve with s = " + str(current_s) +
                    " from the sequence " + str(ss))
            
            if attempt == attempts[-1]:
                
                break
            
            if current_s >= maxval:
            
                print("Exceeded maximum regularization (s_max = " + 
                    str(maxval) + ")")
                
                break
            
            index = ss.index(current_s)
            
            if index == 0:
            
                s_to_insert = 2.*ss[0]
                
                new_ss = (s_to_insert,) + ss
                
                model.solution.assign(model.solutions[-1])
            
            else:
            
                s_to_insert = (current_s + ss[index - 1])/2.
            
                new_ss = ss[:index] + (s_to_insert,) + ss[index:]
                
                model.solution.assign(model.backup_solution)
            
            smoothing_sequence = new_ss
            
            if not model.quiet:
            
                print("Inserted new value of " + str(s_to_insert))
            
            first_s_to_solve = s_to_insert
    
    assert(solved)
    
    assert(model.smoothing.__float__() ==
        smoothing_sequence[-1])
        
    model.smoothing_sequence = smoothing_sequence
    