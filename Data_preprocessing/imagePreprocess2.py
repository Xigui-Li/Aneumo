"""
imagePreprocess2.py

This script applies a series of transformations to 3D medical images, including 
cropping, padding, and optional scaling coordination. It leverages MONAI for 
medical image preprocessing and PyTorch for tensor manipulations. The code 
demonstrates how to crop images based on foreground information, apply scaling 
factors, and save the processed outputs for downstream tasks.

Usage:
    python imagePreprocess2.py
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

# define model
class CropImageByScaledCoord():
    """
    Custom class for cropping and padding 3D images with optional scaling of the bounding box.
    It can automatically ensure that the cropped region meets a minimum size threshold.

    Args:
        scale (int): Scaling factor for the bounding box. Default is 2.
        min_x (int): Minimum size in the X dimension. Default is 32.
        min_y (int): Minimum size in the Y dimension. Default is 32.
        min_z (int): Minimum size in the Z dimension. Default is 32.
        lazy (bool): Indicator for lazy transforms. Default is False.
        mode (str): Padding mode, usually from PyTorch pad modes. Default is "constant".
    """
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
        """
        Enable or disable lazy transforms for padding and cropping operations.
        """
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
        Compute and adjust the bounding box coordinates to ensure divisibility by `k_divisible`.

        Args:
            img (torch.Tensor): The input 3D volume.
            init_start (np.ndarray): Initial start coordinate for bounding box.
            init_end (np.ndarray): Initial end coordinate for bounding box.

        Returns:
            tuple: The final bounding box start and end coordinates.
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
        Crop and pad the input image based on the provided bounding box coordinates.

        Args:
            img (torch.Tensor): The input 3D volume.
            box_start (np.ndarray): Start coordinate for the bounding box.
            box_end (np.ndarray): End coordinate for the bounding box.
            mode (str, optional): Padding mode. Default is None (use class-level mode).
            lazy (bool, optional): Whether to apply lazy transforms. 
            **pad_kwargs: Additional arguments passed to the padder.

        Returns:
            torch.Tensor: The cropped and padded 3D volume.
        """
        # Create slices for cropping
        slices = self.compute_slices(roi_start=box_start, roi_end=box_end)
        cropped = Crop().__call__(img=img, slices=slices, lazy=lazy)
        
        # Determine how much to pad in each dimension
        pad_to_start = np.maximum(-box_start, 0)
        pad_to_end = np.maximum(
            box_end - np.asarray(img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]), 0
        )
        pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
        pad_width = BorderPad(spatial_border=pad).compute_pad_width(
            cropped.peek_pending_shape() if isinstance(cropped, MetaTensor) else cropped.shape[1:]
        )

        # Apply padding
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
        """
        Execute the crop and pad pipeline on the provided 3D image and bounding box info.

        Args:
            x_in (dict): A dictionary containing:
                - "image": The input 3D volume (torch.Tensor).
                - "foreground_start_coord": NumPy array for bounding box start.
                - "foreground_end_coord": NumPy array for bounding box end.

        Returns:
            dict: The modified input with updated 'image' and bounding box information.
        """        
        _,H,W,Z=x_in["image"].shape
        start_coord = x_in['foreground_start_coord']
        end_coord = x_in['foreground_end_coord']
        box_start, box_end = self.compute_bounding_box(x_in['image'],start_coord,end_coord)
        x_in['foreground_start_coord'] = start_coord
        x_in['foreground_end_coord'] = box_end
        x_in['image'] = self.crop_pad(x_in['image'], box_start, box_end,mode=self.mode)
      
        return x_in



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




# corp image

raw_image_path = 'data/niidata/'
raw_label_path = 'data/niidata/'
#corped_image_path = 'data/image/'
corped_image_path = 'data/image_5/'

if not os.path.exists(corped_image_path):
    os.makedirs(corped_image_path)

# Process each case in the specified range
for case_id in tqdm(range(1, 10654), desc="Processing images", ncols=100, unit="image"):
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