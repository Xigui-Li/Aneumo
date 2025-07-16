"""
Neural Network Architectures for DeepONet-CFD Project

This module defines the neural network architectures used in the DeepONet-CFD project, 
including the Trunk, Branch, and Branch_Bypass networks. These networks are designed 
to process input features and generate outputs for CFD surrogate modeling tasks.

Classes:
- Trunk: A fully connected neural network for processing trunk inputs.
- Branch: A fully connected neural network for processing branch inputs.
- Branch_Bypass: A simple linear layer for bypassing branch inputs.
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
from monai import transforms
from monai.transforms import Crop,Pad
from monai.transforms.transform import LazyTransform
from collections.abc import Callable, Sequence
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.utils import compute_divisible_spatial_size
from monai.data.meta_tensor import MetaTensor
from itertools import chain
from monai.transforms.croppad.array import BorderPad
from monai.data.meta_obj import get_track_meta
from einops import rearrange
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



torch.set_default_dtype(torch.float32)
device = torch.device("cuda:0")
torch.manual_seed(20231028)

################################################################################


class Trunk(nn.Module):  
    def __init__(self, in_dim=3, out_dim=96, hidden_num=24, layer_num=4):  
        super(Trunk, self).__init__()  
         
        # input layer  
        self.IN = nn.Linear(in_dim, hidden_num)         

        # hidden layers  
        self.hiddens = nn.ModuleList([  
            nn.Linear(hidden_num, hidden_num) for _ in range(layer_num)  
        ])  
          
        # output layer  
        self.OUT = nn.Linear(hidden_num, out_dim)  
  
    def forward(self, x):  
       
        # x = x * 1e3
        
        u = self.IN(x)  
        u = F.gelu(u)  
        
        for layer in self.hiddens:  
            u = layer(u)  
            u = F.gelu(u)  
          
        u = self.OUT(u)
        u = F.gelu(u)
        
        # print('u',u.shape)
        
        c = u.shape[2]
        
        u1 = u[...,0:c//4]
        u2 = u[...,c//4:2*c//4]
        u3 = u[...,2*c//4:3*c//4]
        u4 = u[...,3*c//4:4*c//4]
        
        return u1,u2,u3,u4  
  

class Branch(nn.Module):  
    def __init__(self, in_dim=6, out_dim=96, hidden_num=24, layer_num=4):  
        super(Branch, self).__init__()  
         
        # input layer  
        self.IN = nn.Linear(in_dim, hidden_num)         

        # hidden layers  
        self.hiddens = nn.ModuleList([  
            nn.Linear(hidden_num, hidden_num) for _ in range(layer_num)  
        ])  
          
        # output layer  
        self.OUT = nn.Linear(hidden_num, out_dim)  
  
    def forward(self, x):  
       
        # x = x * 1e3
        
        u = self.IN(x)  
        u = F.gelu(u)  
        
        for layer in self.hiddens:  
            u = layer(u)  
            u = F.gelu(u)  
          
        u = self.OUT(u)
        u = F.gelu(u)
        
        return u

class Branch_Bypass(nn.Module):  
    def __init__(self, in_dim=1, out_dim=4):  
        super(Branch_Bypass, self).__init__()  
         
        # input layer  
        self.FC = nn.Linear(in_dim, out_dim,bias=True)         

  
    def forward(self, x):  
       
        # x = x * 1e3
        
        u = self.FC(x)  
        
        return u    
    


    
class Swin_Final(nn.Module):
    def __init__(self,embed_size):
        super(Swin_Final, self).__init__()
        patch_size = ensure_tuple_rep(2, 3)
        window_size = ensure_tuple_rep(7, 3)
        self.swinViT = SwinViT(
        in_chans=1,
        embed_dim=embed_size,
        window_size=window_size,
        patch_size=patch_size,
        depths=[2,2,2,2],
        num_heads=[2,2,2,2],            
        # num_heads=[3, 6, 12, 24],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=torch.nn.LayerNorm,
        use_checkpoint=False,
        spatial_dims=3,
        use_v2=False
        )
        self.avg_pooling = nn.AdaptiveAvgPool3d((1,1,1))  

    def forward(self, x_in):
        
        hidden_states_out = self.swinViT(x_in)
        out = hidden_states_out[-1]
        out = self.avg_pooling(out)
        
        
        #print(out.shape)
        
        b = out.shape[0]
    
        out = F.normalize(out.reshape(b, -1),p=1,dim=1)

        return out
        

    