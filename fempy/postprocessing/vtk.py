import numpy as np
from matplotlib import pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy


# Trying ideas from https://perso.univ-rennes1.fr/pierre.navaro/read-vtk-with-python.html

def read_vtk_data(vtk_filepath):
    
    reader = vtk.vtkXMLUnstructuredGridReader()

    reader.SetFileName(vtk_filepath)

    reader.Update()
    
    data = reader.GetOutput()
    
    return data
    
    
def get_connectivity(vtk_data):
    
    cell_data = vtk_to_numpy(vtk_data.GetCells().GetData())
    
    connectivity = np.take(
        cell_data,
        [i for i in range(cell_data.size) if i%4 != 0]\
        ).reshape(cell_data.size//4, 3)
        
    return connectivity
    

def plot_mesh(vtk_data):
    """ Plot the mesh """
    connectivity = get_connectivity(vtk_data)
    
    point_data = vtk_to_numpy(vtk_data.GetPoints().GetData())
    
    fig, axes = plt.subplots()
    
    plt.triplot(point_data[:,0], point_data[:,1], connectivity, axes = axes)
    
    axes.set_aspect("equal")
    
    return fig, axes

    
def plot_field_contour(vtk_data, scalar_solution_component = 0, contours = 16):
    """ Plot contours of a scalar field """
    point_data = vtk_to_numpy(vtk_data.GetPoints().GetData())
    
    x = point_data[:,0]
    
    y = point_data[:,1]
    
    u = vtk_to_numpy(
        vtk_data.GetPointData().GetArray(scalar_solution_component))
    
    fig, axes = plt.subplots()
    
    plt.tricontour(x, y, get_connectivity(vtk_data), u, contours, axes = axes)
    
    axes.set_aspect("equal")
    
    plt.colorbar(ax = axes)
    
    plt.xlabel("$x$")
    
    plt.ylabel("$y$")
    
    return fig, axes
    
    
    
def plot_velocity_streamlines(solution_filepath):
    
    # https://matplotlib.org/gallery/images_contours_and_fields/plot_streamplot.html
    assert(False)
    