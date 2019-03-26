import numpy as np
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
    connectivity = np.take(
        cell_data,
        [i for i in range(cell_data.size) if i%4 != 0]\
        ).reshape(cell_data.size//4, 3)
        
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
        contours = 16,
        filled = False,
        axes = None):
    """ Plot contours of a scalar field """
    point_data = vtk_to_numpy(vtk_data.GetPoints().GetData())
    
    x = point_data[:,0]
    
    y = point_data[:,1]
    
    u = vtk_to_numpy(
        vtk_data.GetPointData().GetArray(scalar_solution_component))
    
    if axes is None:
    
        _, axes = plt.subplots()
    
    args = (x, y, get_connectivity(vtk_data), u, contours)
    
    if filled:
    
        mappable = axes.tricontourf(*args)
        
    else:
    
        mappable = axes.tricontour(*args)
    
    axes.set_aspect("equal")
    
    colorbar = plt.colorbar(mappable = mappable, ax = axes)
    
    axes.set_xlabel("$x$")
    
    axes.set_ylabel("$y$")
    
    return axes, colorbar
    
    
def plot_scalar_field(vtk_data, scalar_solution_component = 0, axes = None):
    """ Plot a scalar field """
    return plot_scalar_field_contours(
        vtk_data = vtk_data,
        scalar_solution_component = scalar_solution_component,
        contours = 128,
        filled = True,
        axes = axes)
        
        
def plot_vector_field(vtk_data, vector_solution_component = 0, axes = None, **kwargs):
    """ Plot a vector field """
    point_data = vtk_to_numpy(vtk_data.GetPoints().GetData())
    
    x = point_data[:,0]
    
    y = point_data[:,1]
    
    u = vtk_to_numpy(
        vtk_data.GetPointData().GetArray(vector_solution_component))
    
    if axes is None:
    
        _, axes = plt.subplots()
    
    plt.quiver(x, y, u[:,0], u[:,1], axes = axes, **kwargs)
    
    axes.set_aspect("equal")
    
    axes.set_xlabel("$x$")
    
    axes.set_ylabel("$y$")
    
    return axes
    