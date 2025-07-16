"""
imagePreprocess2-1.py

This script performs 3D image preprocessing, including reading .npy files,
padding them to a specified spatial size, and converting them into NumPy arrays
for downstream processing or training tasks. It leverages multiprocessing for
faster execution.

Usage:
    python imagePreprocess2-1.py
"""
import torch
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
from monai import transforms
from monai.transforms import Crop, Pad
import torch.nn as nn
import math
from monai.transforms.transform import LazyTransform
from collections.abc import Callable, Sequence
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.utils import compute_divisible_spatial_size
from monai.data.meta_tensor import MetaTensor
from itertools import chain
from monai.transforms.croppad.array import BorderPad
from monai.data.meta_obj import get_track_meta
from einops import rearrange
import torch.nn.functional as F
from monai.utils import (
    LazyAttr,
    Method,
    PytorchPadMode,
    TraceKeys,
    TransformBackends,
    convert_data_type,
    convert_to_tensor,
    deprecated_arg_default,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    pytorch_after,
)

import numpy as np
import os
import csv
from tqdm import tqdm
import multiprocessing
from functools import partial

# Define transformations for validation or preprocessing
val_transform_list = [
    transforms.Lambdad(keys=["image", "label"], func=lambda x: np.load(x)[0, :, :, :, :]),
    transforms.SpatialPadd(
        keys=['image', "label"],
        spatial_size=(80, 80, 80),
        method='symmetric',
        mode='constant'
    ),
    transforms.ToTensord(keys=["image"])
]
val_transform = transforms.Compose(val_transform_list)

def process_case(case_id, raw_image_path, raw_label_path, corped_image_path):
    """
    Process a single 3D image case by applying spatial padding and 
    transforming it into a NumPy array.

    Args:
        case_id (int): Unique identifier for the case to be processed.
        raw_image_path (str): Directory containing the raw image .npy files.
        raw_label_path (str): Directory containing the label .npy files.
        corped_image_path (str): Output directory for the preprocessed files.

    Returns:
        tuple:
            case_id (int): The processed case ID.
            success (bool): True if processed successfully, otherwise False.
            error (str or None): Error message if an exception occurred, else None.
    """
    try:
        # Build full file paths
        image_path = os.path.join(raw_image_path, f"{case_id}.npy")
        label_path = os.path.join(raw_label_path, f"{case_id}.npy")
        
       # Apply transformations
        data_dict = val_transform({
            "image": image_path,
            "label": label_path
        })
        
        # Extract image data (remove unnecessary .cuda() calls)
        image_tensor = data_dict["image"][:, None, ...]  # [C, D, H, W]
        image_np = image_tensor.numpy()  
        
        # Save the preprocessed array
        output_path = os.path.join(corped_image_path, f"{case_id}.npy")
        np.save(output_path, image_np)
        
        # Optional logging
        print(f"Processed case {case_id} | Shape: {image_np.shape} | Min: {np.min(image_np)} | Max: {np.max(image_np)}")
        return (case_id, True, None)
    except Exception as e:
        return (case_id, False, str(e))
    except:
        return (case_id, False, "Unknown error")

if __name__ == "__main__":
    # Configure paths
    raw_image_path = 'data/image_5_norma/'
    raw_label_path = 'data/image_5_norma/'
    corped_image_path = 'data/image_5_fixed/'

    # Create output directory if it doesn't exist
    os.makedirs(corped_image_path, exist_ok=True)

    # Generate a list of desired case IDs
    case_ids = list(range(201, 10654))  
    
    # Set the number of worker processes (adjust based on CPU cores)
    num_workers = 16 

   # Create a process pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Fix partial arguments for shared paths
        worker_func = partial(
            process_case,
            raw_image_path=raw_image_path,
            raw_label_path=raw_label_path,
            corped_image_path=corped_image_path
        )
        
        # Use imap_unordered to parallelize tasks more efficiently
        results = pool.imap_unordered(worker_func, case_ids, chunksize=10)
        
        # Initialize progress bar
        with tqdm(total=len(case_ids), desc="Processing", ncols=100, unit="image") as pbar:
            for case_id, success, error in results:
                if not success:
                    print(f"Error processing case {case_id}: {error}")
                pbar.update(1)

    print("All cases processed!")