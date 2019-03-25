import numpy as np
from matplotlib import pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def plot_mesh(solution_filepath):
    """ Plot the mesh """
    # Trying ideas from https://perso.univ-rennes1.fr/pierre.navaro/read-vtk-with-python.html
    reader = vtk.vtkXMLUnstructuredGridReader()

    reader.SetFileName(solution_filepath)

    reader.Update()
    
    reader_output = reader.GetOutput()
    
    
    points = reader_output.GetPoints()
    
    point_data = vtk_to_numpy(points.GetData())
    
    
    cell_data = vtk_to_numpy(reader.GetOutput().GetCells().GetData())
    
    connectivity = np.take(
        cell_data,
        [i for i in range(cell_data.size) if i%4 != 0]\
        ).reshape(cell_data.size//4, 3)
    
    
    fig, axes = plt.subplots()
    
    plt.triplot(point_data[:,0], point_data[:,1], connectivity, axes = axes)
    
    axes.set_aspect("equal")
    
    return fig, axes
