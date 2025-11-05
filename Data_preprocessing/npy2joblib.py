"""
transfer_data2onejob.py

This script reads CFD data (.npz) and associated image data (.npy), applies 
preprocessing steps (e.g., unit conversion, normalization, and region filtering), 
and stores the combined data as a single .job file for each case. It also includes 
a utility function to compare the shapes of original data and the stored data.

Usage:
    python transfer_data2onejob.py
"""
import os
import time
import joblib
import numpy as np
from multiprocessing import Pool, cpu_count

# Directory to store the processed data
output_dir = "job_all"
os.makedirs(output_dir, exist_ok=True)  

# Paths to image and CFD data
img_path = "data/corped_image/"
cfd_path = "data/cfd/"

# Function to compute Euclidean distance
def calDist(x, x_inlet):
    return np.linalg.norm(x - x_inlet, axis=1)

dist_cut = 0.5  # mm

# All data IDs (train + test)
ID_list = list(range(1, 10564))  

# All flow rates
flow_rate_list = ["m=0.001", "m=0.0015", "m=0.002", "m=0.0025",
                  "m=0.003", "m=0.0035", "m=0.00375", "m=0.004"]


def load_and_process_case(args):
    """
    Read and preprocess data for a single (case_id, flow_rate) pair, then store 
    the results in a .job file.

    Args:
        args (tuple): (case_id, flow_rate)

    Returns:
        str: A message indicating success or an error.
    """
    case_id, flow_rate = args
    try:
        print(f"Processing case {case_id} with flow rate {flow_rate}...")

        # Step 1: Build the storage path
        case_dir = os.path.join(output_dir, f"case{case_id}")
        os.makedirs(case_dir, exist_ok=True)  

        # Step 2: Read image data (.npy) and CFD data (.npz)
        image_data = np.load(os.path.join(img_path, f"{case_id}.npy"))
        cfd_data = np.load(os.path.join(cfd_path, flow_rate, f"{case_id}.npz"))

        # Step 3: Preprocess CFD data
        x_temp = cfd_data["X_sup"][0, ...] * 1000  
        y_temp = cfd_data["Y_sup"][0, ...]
        y_temp[:, 0] -= np.mean(y_temp[:, 0])  
        x_img_temp = image_data
        x_in_temp = cfd_data["Simple_inlet"]
        x_img_temp = (x_img_temp - np.min(x_img_temp)) / (np.max(x_img_temp) - np.min(x_img_temp))  # 归一化

        x_inlet_temp = cfd_data["X_inlet"][0, ...]

        # Step 3.1: Filter out points too close to the inlet
        mask = (x_temp[:, 0] != 1e6)
        for i in range(len(x_temp)):
            dist_min = np.min(calDist(x_temp[i, 0:3], x_inlet_temp))
            if dist_min < dist_cut:
                mask[i] = False

        x_temp = x_temp[mask, :]
        y_temp = y_temp[mask, :]

        # Step 3.2: Correct the shape of x_in_temp
        x_in_temp = np.array(x_in_temp)  
        if len(x_in_temp.shape) == 3 and x_in_temp.shape[1] == 1:
            x_in_temp = x_in_temp.squeeze(1)  # (B, 1, 7) → (B, 7)
        elif len(x_in_temp.shape) == 2:
            pass  
        elif len(x_in_temp.shape) == 3 and x_in_temp.shape[0] == 1:
            x_in_temp = x_in_temp.squeeze(0)  # (1, 7) → (7)
        elif len(x_in_temp.shape) == 1 and x_in_temp.shape[0] == 7:
            x_in_temp = np.expand_dims(x_in_temp, axis=0)  # (7,) → (1, 7)
        else:
            raise ValueError(f"Unexpected x_in_temp shape before saving: {x_in_temp.shape}, expected (B, 7)")

        # Step 4: Store data into a single .job file
        combined_data = {
            "x_temp": x_temp,
            "y_temp": y_temp,
            "x_img_temp": x_img_temp,
            "x_in_temp": x_in_temp
        }
        joblib.dump(combined_data, os.path.join(case_dir, f"{flow_rate.replace('=', '')}.job"))

        return f"Successfully processed case {case_id} with flow rate {flow_rate}"

    except Exception as e:
        return f"Error processing case {case_id} with flow rate {flow_rate}: {e}"


# 并行处理所有数据
def process_all_data_parallel():
    """
    Parallelize the data loading and processing for all (case_id, flow_rate) 
    pairs using multiprocessing.
    """
    start_time = time.time()
    print(f"Starting parallel processing with {cpu_count()} cores...")

    # Generate all combinations of case_id and flow_rate
    all_cases = [(case_id, flow_rate) for flow_rate in flow_rate_list for case_id in ID_list]

    # Use a process pool to parallelize the work
    with Pool(processes=cpu_count() // 2) as pool:  
        results = pool.map(load_and_process_case, all_cases)

    for res in results:
        print(res)

    end_time = time.time()
    print(f"Finished processing all data in {end_time - start_time:.2f} s")

# Invoke the parallel data processing
process_all_data_parallel()

def compare_shapes():
    """
    Compare the shapes of the original data versus the stored data 
    to ensure consistency.
    """
    print("\n=== Comparing Original Data vs. Stored Data ===")

    # Read a sample of the original data
    original_sample = np.load(os.path.join(img_path, "1.npy"))
    original_cfd = np.load(os.path.join(cfd_path, "m=0.001", "1.npz"))
    
    original_x_temp = original_cfd["X_sup"][0, ...] * 1000
    original_y_temp = original_cfd["Y_sup"][0, ...]
    original_x_img_temp = original_sample
    original_x_in_temp = original_cfd["Simple_inlet"]

    # Read the stored .job file
    stored_x_temp = joblib.load(os.path.join(output_dir, "case1/m0.001.job"))["x_temp"]
    stored_y_temp = joblib.load(os.path.join(output_dir, "case1/m0.001.job"))["y_temp"]
    stored_x_img_temp = joblib.load(os.path.join(output_dir, "case1/m0.001.job"))["x_img_temp"]
    stored_x_in_temp = joblib.load(os.path.join(output_dir, "case1/m0.001.job"))["x_in_temp"]

    print(f"Original x_temp shape: {original_x_temp.shape}, Stored x_temp shape: {stored_x_temp.shape}")
    print(f"Original y_temp shape: {original_y_temp.shape}, Stored y_temp shape: {stored_y_temp.shape}")
    print(f"Original x_img_temp shape: {original_x_img_temp.shape}, Stored x_img_temp shape: {stored_x_img_temp.shape}")
    print(f"Original x_in_temp shape: {original_x_in_temp.shape}, Stored x_in_temp shape: {stored_x_in_temp.shape}")

# Compare shapes to verify correctness
compare_shapes()
