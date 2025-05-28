import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


torch.manual_seed(20231028)
np.random.seed(20231028)
torch.cuda.manual_seed(20231028)

device = 'cuda:0'


################################################################
# Loss
################################################################

class DataLoss_Bypass_Plus(nn.Module):
    def __init__(self, trunk_net,branch_img_net,branch_bc_net,branch_bp_net):
        super(DataLoss_Bypass_Plus,self).__init__()
        self.trunk = trunk_net
        self.branch_img = branch_img_net
        self.branch_bc = branch_bc_net
        self.branch_bp = branch_bp_net

    
    def forward(self, x, y_true, image, bc):
        
        
        trunk_pred1,trunk_pred2,trunk_pred3,trunk_pred4 = self.trunk(x) # shape 4*[B,Np,Nm]
        branch_img_pred = self.branch_img(image).unsqueeze(-1) # shape [B,Nimg,1]
        branch_bc_pred = self.branch_bc(bc).unsqueeze(-1) # shape [B,Nbc,1]
        # if len(branch_bc_pred.shape) == 5 and branch_bc_pred.shape[0] ==1:
        #     branch_bc_pred = branch_bc_pred.squeeze(0)
        #     branch_bc_pred = branch_bc_pred.squeeze(1)
        # print(f"img:{branch_img_pred.shape},bc:{branch_bc_pred.shape}")
        branch_pred = torch.cat([branch_img_pred,branch_bc_pred],dim=1) # shape [B,Nm,1], Nm = Nimg + Nbc
        
        # print(f"img:{trunk_pred1.shape},bc:{branch_pred.shape}")

        h_pred1 = torch.matmul(trunk_pred1,branch_pred) # shape [B,Np,1]
        h_pred2 = torch.matmul(trunk_pred2,branch_pred) # shape [B,Np,1]
        h_pred3 = torch.matmul(trunk_pred3,branch_pred) # shape [B,Np,1]
        h_pred4 = torch.matmul(trunk_pred4,branch_pred) # shape [B,Np,1]
        
        y_pred = torch.cat([h_pred1,h_pred2,h_pred3,h_pred4],dim=-1) # shape [B,Np,4]
        
        branch_bp_pred = self.branch_bp(bc[...,-1:]) # shape [B,4]
        # print(f"y_pred: {y_pred.shape}")
        # print(f"branch_bp_pred:{branch_bp_pred.shape}")
        y_pred =  y_pred * branch_bp_pred.unsqueeze(1) # shape [B,Np,4]
        
        p_ref_pred = torch.mean(y_pred[...,0:1],dim=1) 
        p_ref_true = torch.mean(y_true[...,0:1],dim=1) 
        
        p_range_pred = torch.max(y_pred[...,0:1],dim=1)[0] - torch.min(y_pred[...,0:1],dim=1)[0]
        p_range_true = torch.max(y_true[...,0:1],dim=1)[0] - torch.min(y_true[...,0:1],dim=1)[0] 
        
        u_range_true = torch.max(y_true[...,1:2],dim=1)[0] - torch.min(y_true[...,1:2],dim=1)[0]
        v_range_true = torch.max(y_true[...,2:3],dim=1)[0] - torch.min(y_true[...,2:3],dim=1)[0]
        w_range_true = torch.max(y_true[...,3:4],dim=1)[0] - torch.min(y_true[...,3:4],dim=1)[0]        
        
        output_pre = torch.sum(torch.square(y_pred[...,0] - p_ref_pred + p_ref_true - y_true[...,0])) 
        output_vel = torch.sum(torch.square(y_pred[...,1:] - y_true[...,1:]))      
        
        output_percentage_pre = torch.sqrt(torch.sum(torch.square(y_pred[...,0] - p_ref_pred + p_ref_true - y_true[...,0])) / torch.sum(torch.square(y_true[...,0]-p_ref_true)))
        output_percentage_u = torch.sqrt(torch.sum(torch.square(y_pred[...,1:2] - y_true[...,1:2])) / torch.sum(torch.square(y_true[...,1:2])))
        output_percentage_v = torch.sqrt(torch.sum(torch.square(y_pred[...,2:3] - y_true[...,2:3])) / torch.sum(torch.square(y_true[...,2:3])))
        output_percentage_w = torch.sqrt(torch.sum(torch.square(y_pred[...,3:4] - y_true[...,3:4])) / torch.sum(torch.square(y_true[...,3:4])))
        
        output_MNAE_pre = torch.mean(torch.abs(y_pred[...,0] - p_ref_pred + p_ref_true - y_true[...,0])/ p_range_true)
        output_MNAE_u = torch.mean(torch.abs(y_pred[...,1] - y_true[...,1])/ u_range_true)
        output_MNAE_v = torch.mean(torch.abs(y_pred[...,2] - y_true[...,2])/ v_range_true)
        output_MNAE_w = torch.mean(torch.abs(y_pred[...,3] - y_true[...,3])/ w_range_true)
        
        output_dp = torch.sqrt(torch.sum(torch.square(p_range_pred-p_range_true)) / torch.sum(torch.square(p_range_true)))
        
        
        return output_pre, output_vel, output_percentage_pre, output_percentage_u, output_percentage_v, output_percentage_w, output_MNAE_pre, output_MNAE_u,  output_MNAE_v, output_MNAE_w, output_dp
    
