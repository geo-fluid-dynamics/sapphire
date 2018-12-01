""" These contents will ideally be obsoleted by upstream updates """


def plot_unit_interval(u_h, u_m, sample_size = 100):

    mesh = u_h.function_space().mesh()
    
    assert(type(mesh) == type(fe.UnitIntervalMesh(1)))
    
    sample_points = [x/float(sample_size) for x in range(sample_size + 1)]
    
    fig = plt.figure()
    
    axes = plt.axes()
    
    plt.plot(
        sample_points, 
        [u_h((p,)) for p in sample_points],
        axes = axes,
        color = "red")
    
    _u_m = fe.interpolate(u_m, u_h.function_space())
    
    axes = plt.plot(
        sample_points, 
        [_u_m((p,)) for p in sample_points],
        axes = axes,
        color = "blue")
    
    plt.axis("square")
    
    plt.xlim((-0.1, 1.1))
    
    plt.legend((r"$u_h$", r"$u_m$"))
    
    plt.xlabel(r"$x$")
    
    plt.ylabel(r"$u$")
    