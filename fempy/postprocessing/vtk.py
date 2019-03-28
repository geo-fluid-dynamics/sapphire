import numpy
import scipy.interpolate
from matplotlib import pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def read_vtk_data(vtk_filepath):
    
    reader = vtk.vtkXMLUnstructuredGridReader()

    reader.SetFileName(vtk_filepath)

    reader.Update()
    
    data = reader.GetOutput()
    
    return data
    
    
def get_connectivity(vtk_data):
    
    cell_data = vtk_to_numpy(vtk_data.GetCells().GetData())
    
    # See https://perso.univ-rennes1.fr/pierre.navaro/read-vtk-with-python.html
    connectivity = numpy.take(
        cell_data,
        [i for i in range(cell_data.size) if i%4 != 0])\
        .reshape(cell_data.size//4, 3)
        
    return connectivity
    

def plot_mesh(vtk_data, axes = None):
    """ Plot the mesh """
    connectivity = get_connectivity(vtk_data)
    
    point_data = vtk_to_numpy(vtk_data.GetPoints().GetData())
    
    if axes is None:
    
        _, axes = plt.subplots()
    
    plt.triplot(point_data[:,0], point_data[:,1], connectivity, axes = axes)
    
    axes.set_aspect("equal")
    
    return axes

    
def plot_scalar_field_contours(
        vtk_data,
        scalar_solution_component = 0,
        filled = False,
        colorbar = True,
        axes = None,
        **kwargs):
    """ Plot contours of a scalar field """
    point_data = vtk_to_numpy(vtk_data.GetPoints().GetData())
    
    x = point_data[:,0]
    
    y = point_data[:,1]
    
    u = vtk_to_numpy(vtk_data.GetPointData().GetArray(
        scalar_solution_component))
    
    if axes is None:
    
        _, axes = plt.subplots()
    
    args = (x, y, get_connectivity(vtk_data), u)
    
    if filled:
    
        mappable = axes.tricontourf(*args, **kwargs)
        
    else:
    
        mappable = axes.tricontour(*args, **kwargs)
    
    axes.set_aspect("equal")
    
    axes.set_xlabel("$x$")
    
    axes.set_ylabel("$y$")
    
    if colorbar:
    
        return axes, plt.colorbar(mappable = mappable, ax = axes)
    
    else:
    
        return axes
        
        
def plot_scalar_field(vtk_data, scalar_solution_component = 0, axes = None):
    """ Plot a scalar field """
    return plot_scalar_field_contours(
        vtk_data = vtk_data,
        scalar_solution_component = scalar_solution_component,
        filled = True,
        axes = axes,
        levels = 128)
        
        
def plot_vector_field(vtk_data, vector_solution_component = 0, axes = None,
        **kwargs):
    """ Plot a vector field """
    point_data = vtk_to_numpy(vtk_data.GetPoints().GetData())
    
    x = point_data[:,0]
    
    y = point_data[:,1]
    
    u = vtk_to_numpy(vtk_data.GetPointData().GetArray(
        vector_solution_component))
    
    if axes is None:
    
        _, axes = plt.subplots()
    
    plt.quiver(x, y, u[:,0], u[:,1], axes = axes, **kwargs)
    
    axes.set_aspect("equal")
    
    axes.set_xlabel("$x$")
    
    axes.set_ylabel("$y$")
    
    return axes
    
    
def format_data_for_streamplot(x, y, u, v):
    
    x_unique = numpy.unique(x)
    
    y_unique = numpy.unique(y)
    
    x_grid, y_grid = numpy.meshgrid(x_unique, y_unique)
    
    u_grid = scipy.interpolate.griddata((x, y), u, (x_grid, y_grid))
    
    v_grid = scipy.interpolate.griddata((x, y), v, (x_grid, y_grid))
    
    return x_unique, y_unique, u_grid, v_grid
    
    
def plot_streamlines(vtk_data, vector_solution_component = 0, axes = None,
        max_linewidth = 3.,
        **kwargs):
    """ Plot streamlines of a vector field """
    points = vtk_to_numpy(vtk_data.GetPoints().GetData())[:,:2]
    
    velocities = vtk_to_numpy(vtk_data.GetPointData().GetArray(
        vector_solution_component))
    
    if axes is None:
    
        _, axes = plt.subplots()
    
    u = velocities[:,0]
    
    v = velocities[:,1]
    
    x_unique, y_unique, u_grid, v_grid = format_data_for_streamplot(
        x = points[:,0],
        y = points[:,1],
        u = u,
        v = v)
        
    speed_grid = numpy.sqrt(u_grid**2 + v_grid**2)
        
    stream = axes.streamplot(x_unique, y_unique, u_grid, v_grid,
        linewidth = max_linewidth*speed_grid/speed_grid.max(),
        arrowsize = 1.e-8,
        **kwargs)
    
    axes.set_aspect("equal")
    
    axes.set_xlabel("$x$")
    
    axes.set_ylabel("$y$")
    
    return axes, stream
    