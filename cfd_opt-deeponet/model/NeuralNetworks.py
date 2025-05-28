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


from collections.abc import Callable, Sequence

from itertools import chain

from einops import rearrange



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
    
    