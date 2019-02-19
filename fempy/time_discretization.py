""" Time discretization formulas """


def bdf(ys, order, timestep_size):
    """ Backward difference formulas """
    assert(len(ys) == (order + 1))
    
    if order == 1:
        
        alphas = (1., -1.)
        
    elif order == 2:
    
        alphas = (3./2., -2., 1./2.)
    
    elif order == 3:
    
        alphas = (11./6., -3., 3./2., -1./3.)
        
    else:
    
        raise(NotImplementedError())
    
    y_t = alphas[-1]*ys[-1]
    
    for alpha, y in zip(alphas[:-1], ys[:-1]):
    
        y_t += alpha*y
    
    y_t /= timestep_size
    
    return y_t
    