"""
cfdPreprocess_pred.py

This script processes CFD data for prediction by loading and transforming existing .npy files, 
computing inlet-centered normals, and preparing input/output tensors for downstream inference tasks. 
It also saves processed files for different flow rates.

Usage:
    python cfdPreprocess_pred.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import csv
from scipy.spatial import KDTree

device = 'cpu'

class Read_Data(nn.Module):
    """
    Reads and processes CFD data from NumPy files, then computes inlet normals.

    Args:
        cfd_path (str): Base path where CFD .npy files are stored.
    """
    def __init__(self, cfd_path):
        super(Read_Data,self).__init__()
        
        self.cfd_path = cfd_path

    def compute_normals(self,points, k=10):
        """
        Compute normal vectors for each point by analyzing its local neighborhood.

        Args:
            points (ndarray): Array of point coordinates.
            k (int): Number of neighbors to consider.

        Returns:
            tuple: (centre_points, centre_normals) 
                   The mean point and averaged normal over the entire set.
        """
        tree = KDTree(points)
        _, indices = tree.query(points, k=k+1)  
        normals = []

        for i in range(len(points)):
           
            neighborhood = points[indices[i]]
            centered_points = neighborhood - np.mean(neighborhood, axis=0)
            # Compute covariance to find principal directions
            cov_matrix = np.cov(centered_points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            normal = eigenvectors[:, 0] # eigenvector with smallest eigenvalue
            normal /= np.linalg.norm(normal)
            normals.append(normal)

        point_normals = np.array(normals)
        centre_points = np.mean(points,axis=0)
        centre_normals = np.mean(normals,axis=0)

        return centre_points, centre_normals
    
    def forward(self,case_id,fsup):
        """
        Load internal, SDF, and inlet arrays, compute inlet center/normal, 
        and return packaged tensors for further processing.

        Args:
            case_id (int): Case ID number for which data is to be processed.
            fsup (float): Placeholder for future scaling factor usage.

        Returns:
            tuple: (X_sup, Y_sup, X_inlet, Simple_inlet)
                   Torch tensors ready for inference or other usage.
        """        
        i = case_id 

        array_data_internal = np.load(self.cfd_path+'array_internal_'+str(i)+'.npy') # xyz puvw
        array_data_sdf = np.load(self.cfd_path+'array_sdf_'+str(i)+'.npy') # xyz sdf puvw
        array_data_inlet = np.load(self.cfd_path+'array_inlet_'+str(i)+'.npy') # xyz puvw
            
        # shift u-velocity to -0.5
        array_data_internal[:,4] = array_data_internal[:,4] - 0.5

        # prepare input and output data    
        x_internal = array_data_internal[:,0:3]
        sdf_internal = array_data_sdf[:,3:4]
        y_internal = array_data_internal[:,3:]
        x_inlet = array_data_inlet[:,0:3]
        
        # calculate inlet centre and normal
        centre_inlet, normal_inlet = self.compute_normals(x_inlet,10)
        simple_inlet = np.concatenate([centre_inlet,normal_inlet],axis=-1)

        # Merge internal XYZ and SDF
        x_sup = np.concatenate([x_internal,sdf_internal],axis=-1)

        # Convert to Torch tensors
        X_sup = torch.tensor(x_sup, dtype=torch.float32, device=device).unsqueeze(0)
        Y_sup = torch.tensor(y_internal, dtype=torch.float32, device=device).unsqueeze(0)
        X_inlet = torch.tensor(x_inlet, dtype=torch.float32, device=device).unsqueeze(0)
        Simple_inlet = torch.tensor(simple_inlet, dtype=torch.float32, device=device).unsqueeze(0)
        
        return X_sup, Y_sup, X_inlet, Simple_inlet

# Process each flow rate
for flow_rate in [
    'm=0.001', 'm=0.0015', 'm=0.002', 'm=0.003', 
    'm=0.0035', 'm=0.004'
]:
    raw_cfd_path = 'data/npydata/'+flow_rate+'/'
    DataLoader = Read_Data(raw_cfd_path)
    processed_cfd_path = 'data/cfd_pred/'+flow_rate+'/'

    # Map flow_rate to numerical representation
    if flow_rate == 'm=0.001':
        Flow_Rate = 0.1
    if flow_rate == 'm=0.0015':
        Flow_Rate = 0.15
    if flow_rate == 'm=0.002':
        Flow_Rate = 0.2
    if flow_rate == 'm=0.0025':
        Flow_Rate = 0.25
    if flow_rate == 'm=0.003':
        Flow_Rate = 0.3
    if flow_rate == 'm=0.0035':
        Flow_Rate = 0.35
    if flow_rate == 'm=0.00375':
        Flow_Rate = 0.375
    if flow_rate == 'm=0.004':
        Flow_Rate = 0.4
        
        
    for case_id in range(3990, 4032):
        
        try:
            X_sup, Y_sup, X_inlet, Simple_inlet = DataLoader(case_id,1.0)

            # Create an array [center_x, center_y, center_z, nx, ny, nz, flow_rate]
            Simple_inlet_ = np.zeros([1,7])
            Simple_inlet_[0,0:6] = Simple_inlet
            Simple_inlet_[0,6] = Flow_Rate

            # Save as .npz file for later inference usage
            np.savez(
                processed_cfd_path + str(case_id) + '.npz',
                X_sup=X_sup,
                Y_sup=Y_sup,
                X_inlet=X_inlet,
                Simple_inlet=Simple_inlet_
            )
            print(case_id,Y_sup.shape,Simple_inlet_.shape)
            
        except:
            1+1