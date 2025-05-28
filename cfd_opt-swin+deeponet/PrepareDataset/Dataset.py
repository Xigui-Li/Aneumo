"""
CFD Dataset for DeepONet-CFD Project

This module defines the `CFDJobDataset` class, which loads CFD simulation data
from `.job` files. It supports random sampling of data points and processes
input features and labels for training.

Classes:
- CFDJobDataset: A PyTorch Dataset for loading CFD simulation data.
"""
import os
import joblib
import glob
import numpy as np
from torch.utils.data import Dataset
import torch

class CFDJobDataset(Dataset):
    def __init__(self, data_dir, case_ids, flow_rates, num_sample=2000):
        """
        CFD Dataset that supports loading .job files by case ID and flow rate,
        and performs random sampling of 2000 points from x_temp and y_temp.

        :param data_dir: The directory where the data files are located
        :param case_ids: List of case IDs to load
        :param flow_rates: List of flow rates (used to match file names)
        :param num_sample: Number of points to sample (default is 2000)
        """
        self.data_dir = data_dir
        self.file_paths = []
        self.num_sample = num_sample  # Fixed number of sampling points is 2000

        # Iterate through all case IDs
        for case_id in case_ids:
            case_folder = f"case{case_id}"
            case_path = os.path.join(data_dir, case_folder)

            if not os.path.isdir(case_path):
                continue  # Skip if the case directory does not exist

            # **Match .job files that correspond to the specified flow rates**
            for flow_rate in flow_rates:

                all_files = glob.glob(os.path.join(case_path, f"{flow_rate}*.job"))

                exact_files = [f for f in all_files if os.path.basename(f) == f"{flow_rate}.job"]
                if exact_files:
                    self.file_paths.extend(exact_files)
                else:
                    print(f"Warning: No exact match for {flow_rate}.job in {case_path}, falling back to pattern matching.")
                    self.file_paths.extend(all_files)
            
        if len(self.file_paths) == 0:
            print(f"Warning: No .job files found in {data_dir} matching the given criteria.")
        else:
            print(f"Loaded {len(self.file_paths)} job files from {data_dir}")

    def __len__(self):
        """Return the size of the dataset"""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Load a .job file's data and perform fixed 2000 point sampling"""
        job_file = self.file_paths[idx]
        stored_data = joblib.load(job_file)

        x_temp = np.array(stored_data["x_temp"])  # shape: (N, D)
        y_temp = np.array(stored_data["y_temp"])  # shape: (N, D)
        x_img_temp = np.array(stored_data["x_img_temp"])  #  (1, 1, C, H, W)
        x_in_temp = np.array(stored_data["x_in_temp"])  #  (1, 7)
        # **Fix x_in_temp shape if necessary*
        if len(x_img_temp.shape) == 3:
            x_img_temp = np.expand_dims(x_img_temp, axis=0)
        elif len(x_img_temp.shape) == 5:
            x_img_temp = x_img_temp.squeeze(0)

        # if len(x_in_temp.shape) == 3 and x_in_temp.shape[1] == 1:
        #     x_in_temp = x_in_temp.squeeze(1)

        # **Perform fixed 2000 point sampling**
        ns = x_temp.shape[0]  # Total number of CFD grid points (N)
        num_sample = min(self.num_sample, ns)  # Ensure the number of points does not exceed actual points
        id_sample = np.random.choice(ns, num_sample, replace=False)  # Randomly select `num_sample` points

        # **Sample x_temp and y_temp**
        x_temp_sampled = x_temp[id_sample, :]  # (num_sample, D)
        y_temp_sampled = y_temp[id_sample, :]  # (num_sample, D)

        # **Convert to PyTorch tensors**
        return (torch.tensor(x_temp_sampled, dtype=torch.float32),
                torch.tensor(y_temp_sampled, dtype=torch.float32),
                torch.tensor(x_img_temp, dtype=torch.float32),
                torch.tensor(x_in_temp, dtype=torch.float32))

