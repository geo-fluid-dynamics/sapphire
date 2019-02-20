""" Time discretization formulas """


def bdf(ys, order, timestep_size):
    """ Backward difference formulas """
    assert(len(ys) == (order + 1))
    
    """ Table of BDF method coefficients """
    if order == 1:
        
        alphas = (1., -1.)
        
    elif order == 2:
    
        alphas = (3./2., -2., 1./2.)
    
    elif order == 3:
    
        alphas = (11./6., -3., 3./2., -1./3.)
        
    elif order == 4:
    
        alphas = (25./12., -4., 3., -4./3., 1./4.)
        
    elif order == 5:
    
        alphas = (137./60., -5., 5., -10./3., 5./4., -1./5.)
        
    elif order == 6:
    
        alphas = (147./60., -60., 15./2., -20./3., 15./4., -6./5., 1./6.)
    
    else: 
    
        raise("BDF is not zero-stable with order > 6.")
        
    
    y_t = alphas[-1]*ys[-1]
    
    for alpha, y in zip(alphas[:-1], ys[:-1]):
    
        y_t += alpha*y
    
    y_t /= timestep_size
    
    
    return y_t
    