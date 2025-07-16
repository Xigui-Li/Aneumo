"""
vtk2npy.py

This script provides two functions to convert VTK-based files (.vtp or .vtu) 
into .npy format. It reads the mesh points, pressure data (p), and velocity 
data (U) from the VTK file, then stores them in a NumPy array of shape (N,7), 
where N is the number of mesh points.

Usage:
    python vtk2npy.py
"""
import vtk
import numpy as np
    
def VTP2NPY(filename):
    """
    Read a .vtp file (stored at filename.vtp), extract p and U arrays, 
    and convert them to a NumPy array of shape (N,7).

    Args:
        filename (str): Path to the .vtp file without the extension.
    
    Returns:
        np.ndarray: With columns [x, y, z, p, u, v, w].
    """
    # Create a vtkXMLPolyDataReader to read the VTP file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename+'.vtp') 
    reader.Update()

    # Retrieve the data object
    polyData = reader.GetOutput()
    points = polyData.GetPoints()
    numPoints = points.GetNumberOfPoints()
    
    # Extract pressure field (p)
    arrayName = 'p'  
    array_p = polyData.GetPointData().GetArray(arrayName)
    # Extract velocity field (U)
    arrayName = 'U' 
    array_U = polyData.GetPointData().GetArray(arrayName)

    # Prepare the output array, each row: [x, y, z, p, u, v, w]
    data = np.zeros([numPoints,7])

    for i in range(numPoints):
        data[i,0:3] = points.GetPoint(i)
        data[i,3] = array_p.GetValue(i)
        data[i,4] = array_U.GetValue(i*3)
        data[i,5] = array_U.GetValue(i*3+1)
        data[i,6] = array_U.GetValue(i*3+2)

    return data

    
def VTU2NPY(filename):
    """
    Read a .vtu file (stored at filename.vtu), extract p and U arrays, 
    and convert them to a NumPy array of shape (N,7).

    Args:
        filename (str): Path to the .vtu file without the extension.
    
    Returns:
        np.ndarray: With columns [x, y, z, p, u, v, w].
    """
    # Create a vtkXMLUnstructuredGridReader to read the VTU file
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename+'.vtu') 
    reader.Update()

    # Retrieve the data object
    polyData = reader.GetOutput()
    points = polyData.GetPoints()
    numPoints = points.GetNumberOfPoints()

    # Extract pressure field (p)
    arrayName = 'p'  
    array_p = polyData.GetPointData().GetArray(arrayName)
    # Extract velocity field (U)
    arrayName = 'U'  
    array_U = polyData.GetPointData().GetArray(arrayName)

    # Prepare the output array, each row: [x, y, z, p, u, v, w]
    data = np.zeros([numPoints,7])

    for i in range(numPoints):
        data[i,0:3] = points.GetPoint(i)
        data[i,3] = array_p.GetValue(i)
        data[i,4] = array_U.GetValue(i*3)
        data[i,5] = array_U.GetValue(i*3+1)
        data[i,6] = array_U.GetValue(i*3+2)

    return data

    
    
    