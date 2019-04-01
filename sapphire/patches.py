""" These contents will ideally be obsoleted by upstream updates """
import firedrake as fe
import matplotlib.pyplot as plt


def plot(u_h, sample_size = 1000, axes = None, color = "k", **kwargs):
    """ fe.plot ignores color argument """
    mesh = u_h.function_space().mesh()
    
    if (type(mesh) == type(fe.UnitIntervalMesh(1))):
        
        sample_points = [x/float(sample_size) for x in range(sample_size + 1)]
        
        if axes is None:
        
            fig = plt.figure()
            
            axes = plt.axes()
    
        plt.plot(
            sample_points, 
            [u_h((p,)) for p in sample_points],
            axes = axes,
            color = color)
    
    else:
    
        fe.plot(u_h, axes = axes, color = color, **kwargs)
    