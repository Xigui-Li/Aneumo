# This script performs data preprocessing for 3D medical images, including cropping,
# padding, normalization, and saving as .npy files. It also supports parallel processing
# to handle a large number of image cases more efficiently. The main goal is to transform
# the images into a consistent format suitable for subsequent model training or inference.

import torch
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
from monai import transforms
from monai.transforms import Crop,Pad
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
import yaml

# define model

class CropImageByScaledCoord():
    def __init__(self,
                scale:int=2,
                min_x:int=32,
                min_y:int=32,
                min_z:int=32,
                lazy:bool=False,
                mode: str = PytorchPadMode.CONSTANT):
        self.scale = scale
        self.min_x,self.min_y,self.min_z=min_x,min_y,min_z
        self.k_divisible=1
        self.padder = Pad(mode=mode, lazy=lazy)
        self.mode = mode
    
    @Crop.lazy.setter  # type: ignore
    def lazy(self, _val: bool):
        self._lazy = _val
        self.padder.lazy = _val

    @staticmethod
    def compute_slices(
        roi_center=None,
        roi_size=None,
        roi_start=None,
        roi_end=None,
        roi_slices=None,
    ):
        """
        Compute the crop slices based on specified `center & size` or `start & end` or `slices`.

        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI, if a dimension of ROI size is larger than image size,
                will not crop that dimension of the image.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI, if a coordinate is out of image,
                use the end coordinate of image.
            roi_slices: list of slices for each of the spatial dimensions.

        """
        roi_start_t: torch.Tensor

        if roi_slices:
            if not all(s.step is None or s.step == 1 for s in roi_slices):
                raise ValueError(f"only slice steps of 1/None are currently supported, got {roi_slices}.")
            return ensure_tuple(roi_slices)
        else:
            if roi_center is not None and roi_size is not None:
                roi_center_t = convert_to_tensor(data=roi_center, dtype=torch.int16, wrap_sequence=True, device="cpu")
                roi_size_t = convert_to_tensor(data=roi_size, dtype=torch.int16, wrap_sequence=True, device="cpu")
                _zeros = torch.zeros_like(roi_center_t)
                half = (
                    torch.divide(roi_size_t, 2, rounding_mode="floor")
                    if pytorch_after(1, 8)
                    else torch.floor_divide(roi_size_t, 2)
                )
                roi_start_t = torch.maximum(roi_center_t - half, _zeros)
                roi_end_t = torch.maximum(roi_start_t + roi_size_t, roi_start_t)
            else:
                if roi_start is None or roi_end is None:
                    raise ValueError("please specify either roi_center, roi_size or roi_start, roi_end.")
                roi_start_t = convert_to_tensor(data=roi_start, dtype=torch.int16, wrap_sequence=True)
                roi_start_t = torch.maximum(roi_start_t, torch.zeros_like(roi_start_t))
                roi_end_t = convert_to_tensor(data=roi_end, dtype=torch.int16, wrap_sequence=True)
                roi_end_t = torch.maximum(roi_end_t, roi_start_t)
            # convert to slices (accounting for 1d)
            if roi_start_t.numel() == 1:
                return ensure_tuple([slice(int(roi_start_t.item()), int(roi_end_t.item()))])
            return ensure_tuple([slice(int(s), int(e)) for s, e in zip(roi_start_t.tolist(), roi_end_t.tolist())])

    def generate_spatial_bounding_box(self,img, init_start:np.array,init_end):
        # scale coord
        center_coord = (init_end+init_start)/2
        new_half_len = np.max(np.stack((np.array(init_end-init_start)*self.scale,np.array((self.min_x,self.min_y,self.min_z)))),axis=0)/2
        # get new coord
        _,H,W,Z = img.shape
        start_coord = center_coord-new_half_len
        start_coord[start_coord<0] = 0
        end_coord = np.min(np.stack((np.array(start_coord+2*new_half_len),img.shape[1:])),axis=0)
        
        return start_coord,end_coord

    def compute_bounding_box(self, img,init_start,init_end):
        """
        Compute the start points and end points of bounding box to crop.
        And adjust bounding box coords to be divisible by `k`.

        """
        box_start, box_end = self.generate_spatial_bounding_box(
            img, init_start,init_end
        )
        box_start_, *_ = convert_data_type(box_start, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        box_end_, *_ = convert_data_type(box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        orig_spatial_size = box_end_ - box_start_
        # make the spatial size divisible by `k`
        spatial_size = np.asarray(compute_divisible_spatial_size(orig_spatial_size.tolist(), k=self.k_divisible))
        # update box_start and box_end
        box_start_ = box_start_ - np.floor_divide(np.asarray(spatial_size) - orig_spatial_size, 2)
        box_end_ = box_start_ + spatial_size
        return box_start_, box_end_

    def crop_pad(
        self,
        img: torch.Tensor,
        box_start: np.ndarray,
        box_end: np.ndarray,
        mode: str = None,
        lazy: bool = False,
        **pad_kwargs,
    ) -> torch.Tensor:
        """
        Crop and pad based on the bounding box.

        """
        slices = self.compute_slices(roi_start=box_start, roi_end=box_end)
        cropped = Crop().__call__(img=img, slices=slices, lazy=lazy)
        pad_to_start = np.maximum(-box_start, 0)
        pad_to_end = np.maximum(
            box_end - np.asarray(img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]), 0
        )
        pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
        pad_width = BorderPad(spatial_border=pad).compute_pad_width(
            cropped.peek_pending_shape() if isinstance(cropped, MetaTensor) else cropped.shape[1:]
        )
        ret = self.padder.__call__(img=cropped, to_pad=pad_width, mode=mode, **pad_kwargs)
        # combine the traced cropping and padding into one transformation
        # by taking the padded info and placing it in a key inside the crop info.
        if get_track_meta() and isinstance(ret, MetaTensor):
            if not lazy:
                ret.applied_operations[-1][TraceKeys.EXTRA_INFO]["pad_info"] = ret.applied_operations.pop()
            else:
                pad_info = ret.pending_operations.pop()
                crop_info = ret.pending_operations.pop()
                extra = crop_info[TraceKeys.EXTRA_INFO]
                extra["pad_info"] = pad_info
                self.push_transform(
                    ret,
                    orig_size=crop_info.get(TraceKeys.ORIG_SIZE),
                    sp_size=pad_info[LazyAttr.SHAPE],
                    affine=crop_info[LazyAttr.AFFINE] @ pad_info[LazyAttr.AFFINE],
                    lazy=lazy,
                    extra_info=extra,
                )
        return ret
    def __call__(self, x_in):
        _,H,W,Z=x_in["image"].shape
        start_coord = x_in['foreground_start_coord']
        end_coord = x_in['foreground_end_coord']
        box_start, box_end = self.compute_bounding_box(x_in['image'],start_coord,end_coord)
        x_in['foreground_start_coord'] = start_coord
        x_in['foreground_end_coord'] = box_end
        x_in['image'] = self.crop_pad(x_in['image'], box_start, box_end,mode=self.mode)
        return x_in

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
    # You should revise the path to your config file
    with open("Preprocess_config.yaml", r, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    scripts = config["scripts"]

    preprocess_ori_config = config["Preprocess_ori"]
    preprocess2_config = config["Preprocess2"]
    preprocess2_1_config = config["Preprocess2_1"]

    if preprocess_ori_config["enable"] == True:
        # Load the original preprocessing configuration
        raw_image_path = preprocess_ori_config["raw_image_path"]
        raw_label_path = preprocess_ori_config["raw_label_path"]
        corped_image_path = preprocess_ori_config["corped_image_path"]

        # Create the output directory if it doesn't exist
        if not os.path.exists(corped_image_path):
            os.makedirs(corped_image_path)
        # Define the transformation pipeline
        val_transform_list = [transforms.LoadImaged(keys=["image","label"]),
                            transforms.EnsureChannelFirstd(keys=["image","label"]),
                            transforms.Orientationd(keys=["image","label"], axcodes="RAS"),
                            transforms.CropForegroundd(keys=["label"],source_key='label',selecct_fn=lambda x:x>0),
                            CropImageByScaledCoord(scale=2,min_x=32,min_y=32,min_z=32),
                            # transforms.SpatialCropD(keys=["image"],roi_start=(0,0,0),roi_end=(96,96,96)),
                            transforms.NormalizeIntensityd(keys=["image"], channel_wise=True),
                            transforms.ToTensord(keys=["image"])]
        val_transform = transforms.Compose(val_transform_list)

        # Process each case in the specified range
        for case_id in range(3900, 4032):
            try:
                image = val_transform({'image':raw_image_path+str(case_id)+'.nii.gz','label':raw_label_path+str(case_id)+'.nii.gz'})['image'][:,None,...].cuda()
                np.save(corped_image_path+str(case_id)+'.npy',image)
                print(case_id,image.shape,np.min(image),np.max(image))
            except:
                1+1

    if preprocess2_config["enable"] == True:
        # Load the original preprocessing configuration
        raw_image_path = preprocess2_config["raw_image_path"]
        raw_label_path = preprocess2_config["raw_label_path"]
        corped_image_path = preprocess2_config["corped_image_path"]

        # Create the output directory if it doesn't exist
        if not os.path.exists(corped_image_path):
            os.makedirs(corped_image_path)

        # Define the transformation pipeline
        val_transform_list = [transforms.LoadImaged(keys=["image","label"]),
                            transforms.EnsureChannelFirstd(keys=["image","label"]),
                            transforms.Orientationd(keys=["image","label"], axcodes="RAS"),
                            transforms.CropForegroundd(keys=["label"],source_key='label',selecct_fn=lambda x:x>0),
                            transforms.Spacingd(keys=["image", "label"], pixdim=(0.18, 0.18, 0.18), mode=("nearest", "nearest")),
                            transforms.SpatialPadd(keys=['image', "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),method='symmetric', mode='constant'),
                            
                            #Median voxel size (x/y/z): [0.03661671 0.03661671 0.03661671]
                            CropImageByScaledCoord(scale=2,min_x=32,min_y=32,min_z=32),
                            # transforms.SpatialCropD(keys=["image"],roi_start=(0,0,0),roi_end=(96,96,96)),
                            # transforms.NormalizeIntensityd(keys=["image"], channel_wise=True),
                            transforms.ToTensord(keys=["image"])]
        val_transform = transforms.Compose(val_transform_list)

        # Process each case in the specified range
        for case_id in tqdm(range(5000, 10654), desc="Processing images", ncols=100, unit="image"):
            try:
                # Load and transform the image
                image = val_transform({'image':raw_image_path+str(case_id)+'.nii.gz','label':raw_label_path+str(case_id)+'.nii.gz'})['image'][:,None,...].cuda()

                # Save the transformed image as .npy
                np.save(corped_image_path + str(case_id) + '.npy', image)
                print(case_id, image.shape, np.min(image), np.max(image))

            except Exception as e:
                print(f"Error processing case {case_id}: {e}")    
            except:
                1+1
    if preprocess2_1_config["enable"] == True:
        raw_image_path = preprocess2_1_config["raw_image_path"]
        raw_label_path = preprocess2_1_config["raw_label_path"]
        corped_image_path = preprocess2_1_config["corped_image_path"]

        # Create output directory if it doesn't exist
        os.makedirs(corped_image_path, exist_ok=True)

    # Generate a list of desired case IDs
    case_ids = list(range(201, 10654))  # 10654-201=10453 cases
    
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
