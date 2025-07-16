"""
npy2vtk.py

This script provides functions to read prediction data from a .npy file and 
apply it to an existing .vtu mesh as new pressure (p) and velocity (U) fields. 
It also supports computing the difference between the predicted and reference 
fields. The resulting data is saved in .vtk format for visualization and 
post-processing with PyVista.

Usage:
    python npy2vtk.py
"""
import pyvista as pv
import numpy as np


def NPY2VTU(npy_filename,ref_filename):
    """
    Load .npy data (XYZ + p + velocity), then read a reference VTU file 
    and update its p and U fields with the predicted values. The mean 
    pressure offset is adjusted to match the reference average pressure.

    Args:
        npy_filename (str): Path without extension to the .npy data file.
        ref_filename (str): Path without extension to the reference .vtu file.
    """
    # Load the .npy data (columns: x, y, z, p, u, v, w)
    data = np.load(npy_filename+'.npy')
    
    # Read the reference VTU file
    mesh = pv.read(ref_filename+'.vtu')
    
    # Extract reference pressure and velocity data
    p = mesh.point_data['p']
    U = mesh.point_data['U']
    
    print('vtk u',U.shape)
    
    # Compute mean pressures for offset adjustment
    p_mean = np.mean(p)
    p_pred_mean = np.mean(data[:,3])

    # Update mesh data for pressure and velocity
    mesh.point_data['p'] = data[:,3] - p_pred_mean + p_mean
    mesh.point_data['U'] = data[:,4:7]

    # Save the modified mesh as .vtk
    mesh.save(npy_filename+'.vtk') 
    
    
def NPY2VTU_diff(npy_filename,ref_filename):
    """
    Load .npy data (XYZ + p + velocity), then read a reference VTU file 
    and update its p and U fields with the difference between predicted 
    and reference values.

    Args:
        npy_filename (str): Path without extension to the .npy data file.
        ref_filename (str): Path without extension to the reference .vtu file.
    """
    # Load the .npy data (columns: x, y, z, p, u, v, w)
    data = np.load(npy_filename+'.npy')
    
    print('data',data.shape)  
    
    # Read the reference VTU file
    mesh = pv.read(ref_filename+'.vtu')
    
    # Extract reference pressure and velocity data
    p = mesh.point_data['p']
    U = mesh.point_data['U']
    
    print('vtk u',U.shape)
    
    # Compute mean pressures for offset
    p_mean = np.mean(p)
    p_pred_mean = np.mean(data[:,3])

    # Update mesh data with difference
    mesh.point_data['p'] = data[:,3] - p_pred_mean - p + p_mean
    mesh.point_data['U'] = data[:,4:7] - U

    # Save the modified mesh as .vtk
    mesh.save(npy_filename+'_diff.vtk') 
    
# Load data 
ID_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Example case IDs
# flow_rate_list = ['m=0.001','m=0.002','m

flow_rate_list = ['m=0.001','m=0.002','m=0.003','m=0.004','m=0.0015', 'm=0.0035']
    
j = 0 
for flow_rate in flow_rate_list:
    for case_id in ID_list:
        source_filename = 'checkpoint/pred_pts_v3/' + str(j)
        base_filename = 'raw_cfd/'+str(case_id)+'/' + flow_rate + '/internal'
        # Apply predicted values to the reference
        NPY2VTU(source_filename,base_filename)
        # Compute and save differences
        NPY2VTU_diff(source_filename,base_filename)
        j = j + 1