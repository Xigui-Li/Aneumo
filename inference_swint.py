import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import cfd_opt_swin_deeponet.model.NeuralNetworks as nn_net
import csv
import time
import numpy as np
import logging
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(20231028)

class CFDDataset(Dataset):
    def __init__(self, cfd_path, img_path, case_ids=None, flow_rates=None):
        """
        Load CFD data (.npz files) and image data (.npy files)
        
        :param cfd_path: Directory containing CFD .npz files
        :param img_path: Directory containing image .npy files
        :param case_ids: List of case IDs to load, if None, load all available cases
        :param flow_rates: List of flow rates to load, if None, load all available flow rates
        """
        self.cfd_path = cfd_path
        self.img_path = img_path
        self.ids = []
        self.flow_rates = []
        self.case_ids = []
        
        if case_ids is None:
            case_ids = [1]
        
        if flow_rates is None:
            flow_rates = ['m=0.002', 'm=0.003']
        
        valid_count = 0
        for case_id in case_ids:
            img_file = os.path.join(img_path, f"{case_id}.npy")
            if not os.path.exists(img_file):
                print(f"Warning: Image file {img_file} does not exist, skipping case {case_id}")
                continue
                
            for flow_rate in flow_rates:
                cfd_file = os.path.join(cfd_path, flow_rate, f"{case_id}.npz")
                if not os.path.exists(cfd_file):
                    print(f"Warning: CFD data file {cfd_file} does not exist, skipping")
                    continue
                
                self.case_ids.append(case_id)
                self.flow_rates.append(flow_rate)
                self.ids.append(valid_count)
                valid_count += 1
        
        print(f"Found {valid_count} valid data pairs")
    
    def __len__(self):
        return len(self.ids)
    
    def calDist(self, x, x_inlet):
        dist1 = x[0] - x_inlet[:,0]
        dist2 = x[1] - x_inlet[:,1]
        dist3 = x[2] - x_inlet[:,2]
        
        dist = np.square(dist1) + np.square(dist2) + np.square(dist3)
        dist = np.sqrt(dist)
        
        return dist
    
    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        flow_rate = self.flow_rates[idx]
        
        image_file = os.path.join(self.img_path, f"{case_id}.npy")
        x_img_temp = np.load(image_file)
        
        # Normalize image data to [0, 1]
        x_img_temp = (x_img_temp - np.min(x_img_temp)) / (np.max(x_img_temp) - np.min(x_img_temp))
        
        # Handle different dimensionality cases
        if len(x_img_temp.shape) == 6:
            x_img_temp = x_img_temp[0]
        
        if len(x_img_temp.shape) != 5:
            # Convert to 5D format if needed
            if len(x_img_temp.shape) == 4:
                x_img_temp = x_img_temp[np.newaxis, ...]
            elif len(x_img_temp.shape) == 3:
                x_img_temp = x_img_temp[np.newaxis, np.newaxis, ...]
                
        if len(x_img_temp.shape) != 5:
            raise ValueError(f"Error: Case {case_id}: Image dimension still not 5D after processing: {x_img_temp.shape}")
        
        cfd_data = np.load(os.path.join(self.cfd_path, flow_rate, f"{case_id}.npz"))
        
        x_temp = cfd_data['X_sup'][0,...] * 1000
        y_temp = cfd_data['Y_sup'][0,...]
        y_temp[:,0] = y_temp[:,0] - np.mean(y_temp[:,0])
        x_in_temp = cfd_data['Simple_inlet']
        
        x_inlet_temp = cfd_data['X_inlet'][0,...]
        
        dist_cut = 0.5
        mask = (x_temp[:,0] != 1e6)
        
        for i in range(len(x_temp)):
            dist = self.calDist(x_temp[i,0:3], x_inlet_temp)
            dist_min = np.min(dist)
            if dist_min < dist_cut:
                mask[i] = False
        
        x_temp = x_temp[mask,:]
        y_temp = y_temp[mask,:]
        
        x_temp = x_temp[np.newaxis,:]
        y_temp = y_temp[np.newaxis,:]
        
        return (torch.tensor(x_temp, dtype=torch.float32),
                torch.tensor(y_temp, dtype=torch.float32),
                torch.tensor(x_img_temp, dtype=torch.float32),
                torch.tensor(x_in_temp, dtype=torch.float32))

def setup_logger(save_folder):
    logger = logging.getLogger("CFD_Inference")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    file_handler = logging.FileHandler(f"{save_folder}/inference.log")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def main(cfd_path, img_path, read_folder, save_folder, case_id_range=None, test_flow_rates=None, use_mixed_precision=True, save_npy=True):
    """
    Perform inference

    :param cfd_path: Directory containing CFD .npz files
    :param img_path: Directory containing image .npy files
    :param read_folder: Model checkpoint folder
    :param save_folder: Results save folder
    :param case_id_range: Tuple (start_id, end_id) specifying case_id range (inclusive), e.g. (1, 1000)
    :param test_flow_rates: List of flow rates to test, if None use all available
    :param use_mixed_precision: Whether to use mixed precision inference
    :param save_npy: Whether to save prediction results as .npy files (default True)
    """
    os.makedirs(save_folder, exist_ok=True)
    logger = setup_logger(save_folder)
    logger.info(f"Starting inference: CFD data={cfd_path}, Image data={img_path}, Model checkpoint={read_folder}, Results={save_folder}, Save npy={save_npy}")

    test_ids = []
    if case_id_range is not None:
        start_id, end_id = case_id_range
        test_ids = list(range(start_id, end_id + 1))
        logger.info(f"Generated case_id range: {start_id} to {end_id}, total {len(test_ids)} case_ids")
    else:
        logger.warning("No case_id_range specified, using default case_ids")
        test_ids = [1]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    precision = torch.float16 if use_mixed_precision else torch.float32
    logger.info(f"Using precision: {precision}")

    test_dataset = CFDDataset(cfd_path, img_path, test_ids, test_flow_rates)
    logger.info(f"Test dataset size: {len(test_dataset)}")

    embed_size = 24
    branch_img_net = nn_net.Swin_Final(embed_size)
    temp_x = torch.zeros([1, 1, 32, 32, 32])
    temp_y = branch_img_net(temp_x)
    logger.info(f'Image network feature size: {temp_y.shape}')

    bc_dim = 64
    branch_bc_net = nn_net.Branch(7, bc_dim, bc_dim, 4)
    in_dim = 4
    hidden_num = temp_y.shape[1] + bc_dim
    out_dim = int(4 * hidden_num)
    layer_num = 4
    branch_bp_net = nn_net.Branch_Bypass(1, 4)
    trunk_net = nn_net.Trunk(in_dim, out_dim, hidden_num, layer_num)

    branch_img_net = branch_img_net.to(device)
    branch_bc_net = branch_bc_net.to(device)
    branch_bp_net = branch_bp_net.to(device)
    trunk_net = trunk_net.to(device)

    checkpoint_id = 5000
    try:
        trunk_state = torch.load(f'{read_folder}/trunk_{checkpoint_id}', map_location=device)
        trunk_state = {k.replace('_orig_mod.', ''): v for k, v in trunk_state.items()}
        trunk_net.load_state_dict(trunk_state)
        logger.info("Successfully loaded trunk_net")

        branch_img_state = torch.load(f'{read_folder}/branch_img_{checkpoint_id}', map_location=device)
        branch_img_state = {k.replace('_orig_mod.', ''): v for k, v in branch_img_state.items()}
        branch_img_net.load_state_dict(branch_img_state)
        logger.info("Successfully loaded branch_img_net")

        branch_bc_state = torch.load(f'{read_folder}/branch_bc_{checkpoint_id}', map_location=device)
        branch_bc_state = {k.replace('_orig_mod.', ''): v for k, v in branch_bc_state.items()}
        branch_bc_net.load_state_dict(branch_bc_state)
        logger.info("Successfully loaded branch_bc_net")

        branch_bp_state = torch.load(f'{read_folder}/branch_bp_{checkpoint_id}', map_location=device)
        branch_bp_state = {k.replace('_orig_mod.', ''): v for k, v in branch_bp_state.items()}
        branch_bp_net.load_state_dict(branch_bp_state)
        logger.info("Successfully loaded branch_bp_net")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return

    with open(os.path.join(save_folder, 'inference_results.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        header = ['case_id', 'flow_rate',
                  'MNAE_pre', 'MNAE_speed', 'MNAE_dp',
                  'MSE_pre', 'MSE_speed', 'MSE_dp',
                  'MAE_pre', 'MAE_speed', 'MAE_dp',
                  'inference_time']
        writer.writerow(header)

    trunk_net.eval()
    branch_img_net.eval()
    branch_bc_net.eval()
    branch_bp_net.eval()

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Inference progress"):
            X, Y, X_img, X_in = test_dataset[idx]
            case_id = test_dataset.case_ids[idx]
            flow_rate = test_dataset.flow_rates[idx]

            X = X.to(device)
            Y = Y.to(device)
            X_img = X_img.to(device)
            X_in = X_in.to(device)
            
            # Handle input dimension adjustment if needed
            if X_in.dim() == 3 and X_in.shape[1] == 1:
                X_in = X_in.squeeze(1)

            pred_time_start = time.time()

            if use_mixed_precision and device.type == 'cuda':
                with autocast():
                    trunk_pred1, trunk_pred2, trunk_pred3, trunk_pred4 = trunk_net(X)
                    branch_img_pred = branch_img_net(X_img).unsqueeze(-1)
                    branch_bc_pred = branch_bc_net(X_in).unsqueeze(-1)

                    branch_pred = torch.cat([branch_img_pred, branch_bc_pred], dim=1)

                    h_pred1 = torch.matmul(trunk_pred1, branch_pred)
                    h_pred2 = torch.matmul(trunk_pred2, branch_pred)
                    h_pred3 = torch.matmul(trunk_pred3, branch_pred)
                    h_pred4 = torch.matmul(trunk_pred4, branch_pred)

                    y_pred = torch.cat([h_pred1, h_pred2, h_pred3, h_pred4], dim=-1)
                    branch_bp_pred = branch_bp_net(X_in[..., -1:])
                    y_pred = y_pred * branch_bp_pred
            else:
                trunk_pred1, trunk_pred2, trunk_pred3, trunk_pred4 = trunk_net(X)
                branch_img_pred = branch_img_net(X_img).unsqueeze(-1)
                branch_bc_pred = branch_bc_net(X_in).unsqueeze(-1)

                branch_pred = torch.cat([branch_img_pred, branch_bc_pred], dim=1)

                h_pred1 = torch.matmul(trunk_pred1, branch_pred)
                h_pred2 = torch.matmul(trunk_pred2, branch_pred)
                h_pred3 = torch.matmul(trunk_pred3, branch_pred)
                h_pred4 = torch.matmul(trunk_pred4, branch_pred)

                y_pred = torch.cat([h_pred1, h_pred2, h_pred3, h_pred4], dim=-1)
                branch_bp_pred = branch_bp_net(X_in[..., -1:])
                y_pred = y_pred * branch_bp_pred

                h_pred1 = torch.matmul(trunk_pred1, branch_pred)
                h_pred2 = torch.matmul(trunk_pred2, branch_pred)
                h_pred3 = torch.matmul(trunk_pred3, branch_pred)
                h_pred4 = torch.matmul(trunk_pred4, branch_pred)

                y_pred = torch.cat([h_pred1, h_pred2, h_pred3, h_pred4], dim=-1)
                branch_bp_pred = branch_bp_net(X_in[..., -1:])
                y_pred = y_pred * branch_bp_pred

            pred_time_end = time.time()
            inference_time = pred_time_end - pred_time_start

            y_true = Y
            
            # Calculate velocity magnitude ||U|| = sqrt(u^2 + v^2 + w^2)
            speed_pred = torch.sqrt(torch.sum(y_pred[..., 1:4]**2, dim=-1, keepdim=True))
            speed_true = torch.sqrt(torch.sum(y_true[..., 1:4]**2, dim=-1, keepdim=True))

            # Reference pressure calculation
            p_ref_pred = torch.mean(y_pred[..., 0:1], dim=1)
            p_ref_true = torch.mean(y_true[..., 0:1], dim=1)

            # Calculate pressure difference (dp = max(p) - min(p))
            p_range_pred = torch.max(y_pred[..., 0:1], dim=1)[0] - torch.min(y_pred[..., 0:1], dim=1)[0]
            p_range_true = torch.max(y_true[..., 0:1], dim=1)[0] - torch.min(y_true[..., 0:1], dim=1)[0]

            # Range calculation for normalization
            p_range_true_norm = p_range_true  # Pressure difference range is p_range_true itself
            speed_range_true = torch.max(speed_true, dim=1)[0] - torch.min(speed_true, dim=1)[0]


            # MNAE (Mean Normalized Absolute Error)
            MNAE_pre = torch.mean(torch.abs(y_pred[..., 0] - p_ref_pred + p_ref_true - y_true[..., 0]) / (p_range_true + 1e-8))
            MNAE_speed = torch.mean(torch.abs(speed_pred - speed_true) / (speed_range_true + 1e-8))
            MNAE_dp = torch.abs(p_range_pred - p_range_true) / (p_range_true_norm + 1e-8)

            # MSE (Mean Squared Error)
            MSE_pre = torch.mean(torch.square(y_pred[..., 0] - p_ref_pred + p_ref_true - y_true[..., 0]))
            MSE_speed = torch.mean(torch.square(speed_pred - speed_true))
            MSE_dp = torch.mean(torch.square(p_range_pred - p_range_true))

            # MAE (Mean Absolute Error)
            MAE_pre = torch.mean(torch.abs(y_pred[..., 0] - p_ref_pred + p_ref_true - y_true[..., 0]))
            MAE_speed = torch.mean(torch.abs(speed_pred - speed_true))
            MAE_dp = torch.mean(torch.abs(p_range_pred - p_range_true))

            # Record results
            result_row = [
                case_id,
                flow_rate,
                float(MNAE_pre.cpu().numpy()),
                float(MNAE_speed.cpu().numpy()),
                float(MNAE_dp.cpu().numpy()),
                float(MSE_pre.cpu().numpy()),
                float(MSE_speed.cpu().numpy()),
                float(MSE_dp.cpu().numpy()),
                float(MAE_pre.cpu().numpy()),
                float(MAE_speed.cpu().numpy()),
                float(MAE_dp.cpu().numpy()),
                inference_time
            ]
            
            with open(os.path.join(save_folder, 'inference_results.csv'), 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(result_row)

            # Save prediction results (if save_npy=True)
            if save_npy:
                save_x = X[0, :, 0:3].cpu().numpy()
                save_p = (y_pred[0, :, 0:1] - torch.mean(y_pred[..., 0:1], dim=1)).cpu().numpy()
                save_U = y_pred[0, :, 1:4].cpu().numpy()
                save_U[:, 0] = save_U[:, 0] + 0.5
                
                save_xy = np.concatenate([save_x, save_p, save_U], axis=-1)
                
                output_file = os.path.join(save_folder, f"{case_id}_{flow_rate.replace('=', '')}.npy")
                np.save(output_file, save_xy)
                
                logger.info(f"Completed inference for case {case_id} flow rate {flow_rate}: shape={save_xy.shape}, time={inference_time:.4f}s, npy file saved")
            else:
                logger.info(f"Completed inference for case {case_id} flow rate {flow_rate}: time={inference_time:.4f}s, npy file not saved")

    logger.info(f"Inference completed! Results saved in {save_folder}")

if __name__ == '__main__':
    cfd_path = "real_data/data/cfd_data"
    img_path = "real_data/data/img_data"
    read_folder = "cfd_opt_swin_deeponet/checkpoint/deeponet_swin"
    save_folder = "predictions/deeponet_swin"
    os.makedirs(save_folder, exist_ok=True)
    case_id_range = (4, 5)
    test_flow_rates = ['m=0.002', 'm=0.003']
    main(cfd_path, img_path, read_folder, save_folder, case_id_range=case_id_range, test_flow_rates=test_flow_rates, use_mixed_precision=False, save_npy=True)
