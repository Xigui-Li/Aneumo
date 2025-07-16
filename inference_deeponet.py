import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import NeuralNetworks as nn_net  # Assumes NeuralNetworks.py contains Trunk, Branch, Branch_Bypass
import csv
import time
import numpy as np
import logging
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(20231028)

def setup_logger(save_folder):
    """
    Configure and return a logger for CFD inference.

    Args:
        save_folder (str): Directory to save the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
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

class CFDDataset(Dataset):
    def __init__(self, cfd_path, case_ids=None, flow_rates=None):
        """
        Initialize the CFD dataset by loading .npz files.

        Args:
            cfd_path (str): Directory containing CFD .npz files.
            case_ids (list, optional): List of case IDs to load. Defaults to [4] if None.
            flow_rates (list, optional): List of flow rates to load. Defaults to ['m=0.002', 'm=0.003'] if None.
        """
        self.cfd_path = cfd_path
        self.ids = []
        self.flow_rates = []
        self.case_ids = []

        if case_ids is None:
            case_ids = [4]

        if flow_rates is None:
            flow_rates = ['m=0.002', 'm=0.003']

        valid_count = 0
        for case_id in case_ids:
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
        """Return the number of data pairs in the dataset."""
        return len(self.ids)

    def calDist(self, x, x_inlet):
        """
        Calculate Euclidean distance between a point and inlet points.

        Args:
            x (np.ndarray): Point coordinates (x, y, z).
            x_inlet (np.ndarray): Inlet point coordinates.

        Returns:
            np.ndarray: Euclidean distances.
        """
        dist1 = x[0] - x_inlet[:,0]
        dist2 = x[1] - x_inlet[:,1]
        dist3 = x[2] - x_inlet[:,2]

        dist = np.square(dist1) + np.square(dist2) + np.square(dist3)
        dist = np.sqrt(dist)

        return dist

    def __getitem__(self, idx):
        """
        Retrieve a data sample by index.

        Args:
            idx (int): Index of the data sample.

        Returns:
            tuple: Tensors for input coordinates, output values, and inlet conditions.
        """
        case_id = self.case_ids[idx]
        flow_rate = self.flow_rates[idx]

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
                torch.tensor(x_in_temp, dtype=torch.float32))

def main(cfd_path, read_folder, save_folder, case_id_range=None, test_flow_rates=None, use_mixed_precision=True, save_npy=True):
    """
    Perform inference using DeepONet without the image branch.

    Args:
        cfd_path (str): Directory containing CFD .npz files.
        read_folder (str): Directory containing model checkpoints.
        save_folder (str): Directory to save inference results.
        case_id_range (list, optional): List of case IDs to process. Defaults to [4] if None.
        test_flow_rates (list, optional): List of flow rates to test. Defaults to all available if None.
        use_mixed_precision (bool): Whether to use mixed precision for inference. Defaults to True.
        save_npy (bool): Whether to save predictions as .npy files. Defaults to True.
    """
    os.makedirs(save_folder, exist_ok=True)
    logger = setup_logger(save_folder)
    logger.info(f"Starting DeepONet inference: CFD data={cfd_path}, model={read_folder}, output={save_folder}, save_npy={save_npy}")

    test_ids = []
    if case_id_range is not None:
        test_ids = case_id_range
        logger.info(f"Using specified case IDs: {test_ids}, total {len(test_ids)} cases")
    else:
        logger.warning("No case_id_range specified, using default case IDs")
        test_ids = [4]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    precision = torch.float16 if use_mixed_precision else torch.float32
    logger.info(f"Using precision: {precision}")

    test_dataset = CFDDataset(cfd_path, test_ids, test_flow_rates)
    logger.info(f"Test dataset size: {len(test_dataset)}")

    bc_dim = 64
    branch_bc_net = nn_net.Branch(7, bc_dim, bc_dim, 4)
    in_dim = 4
    hidden_num = bc_dim
    out_dim = int(4 * hidden_num)
    layer_num = 4
    branch_bp_net = nn_net.Branch_Bypass(1, 4)
    trunk_net = nn_net.Trunk(in_dim, out_dim, hidden_num, layer_num)

    branch_bc_net = branch_bc_net.to(device)
    branch_bp_net = branch_bp_net.to(device)
    trunk_net = trunk_net.to(device)

    checkpoint_id = 5000
    try:
        trunk_state = torch.load(f'{read_folder}/trunk_{checkpoint_id}', map_location=device)
        trunk_state = {k.replace('_orig_mod.', ''): v for k, v in trunk_state.items()}
        trunk_net.load_state_dict(trunk_state)
        logger.info("Successfully loaded trunk_net")

        branch_bc_state = torch.load(f'{read_folder}/branch_bc_{checkpoint_id}', map_location=device)
        branch_bc_state = {k.replace('_orig_mod.', ''): v for k, v in branch_bc_state.items()}
        branch_bc_net.load_state_dict(branch_bc_state)
        logger.info("Successfully loaded branch_bc_net")

        branch_bp_state = torch.load(f'{read_folder}/branch_bp_{checkpoint_id}', map_location=device)
        branch_bp_state = {k.replace('_orig_mod.', ''): v for k, v in branch_bp_state.items()}
        branch_bp_net.load_state_dict(branch_bp_state)
        logger.info("Successfully loaded branch_bp_net")

    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return

    with open(os.path.join(save_folder, 'inference_results_deeponet.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        header = ['case_id', 'flow_rate',
                  'MNAE_pre', 'MNAE_speed', 'MNAE_dp',
                  'MSE_pre', 'MSE_speed', 'MSE_dp',
                  'MAE_pre', 'MAE_speed', 'MAE_dp',
                  'inference_time']
        writer.writerow(header)

    trunk_net.eval()
    branch_bc_net.eval()
    branch_bp_net.eval()

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="DeepONet inference progress"):
            X, Y, X_in = test_dataset[idx]

            case_id = test_dataset.case_ids[idx]
            flow_rate = test_dataset.flow_rates[idx]

            X = X.to(device)
            Y = Y.to(device)
            X_in = X_in.to(device)

            if X_in.dim() == 3 and X_in.shape[1] == 1:
                X_in = X_in.squeeze(1)

            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                pred_time_start = time.time()

            if use_mixed_precision and device.type == 'cuda':
                with autocast():
                    trunk_pred1, trunk_pred2, trunk_pred3, trunk_pred4 = trunk_net(X)
                    branch_bc_pred = branch_bc_net(X_in).unsqueeze(-1)

                    branch_pred = branch_bc_pred

                    h_pred1 = torch.matmul(trunk_pred1, branch_pred)
                    h_pred2 = torch.matmul(trunk_pred2, branch_pred)
                    h_pred3 = torch.matmul(trunk_pred3, branch_pred)
                    h_pred4 = torch.matmul(trunk_pred4, branch_pred)

                    y_pred = torch.cat([h_pred1, h_pred2, h_pred3, h_pred4], dim=-1)
                    branch_bp_pred = branch_bp_net(X_in[..., -1:])
                    y_pred = y_pred * branch_bp_pred.unsqueeze(1)

            else:
                trunk_pred1, trunk_pred2, trunk_pred3, trunk_pred4 = trunk_net(X)
                branch_bc_pred = branch_bc_net(X_in).unsqueeze(-1)

                branch_pred = branch_bc_pred

                h_pred1 = torch.matmul(trunk_pred1, branch_pred)
                h_pred2 = torch.matmul(trunk_pred2, branch_pred)
                h_pred3 = torch.matmul(trunk_pred3, branch_pred)
                h_pred4 = torch.matmul(trunk_pred4, branch_pred)

                y_pred = torch.cat([h_pred1, h_pred2, h_pred3, h_pred4], dim=-1)
                branch_bp_pred = branch_bp_net(X_in[..., -1:])
                y_pred = y_pred * branch_bp_pred.unsqueeze(1)

            if device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                inference_time = start_event.elapsed_time(end_event) / 1000.0
            else:
                pred_time_end = time.time()
                inference_time = pred_time_end - pred_time_start

            y_true = Y

            speed_pred = torch.sqrt(torch.sum(y_pred[..., 1:4]**2, dim=-1, keepdim=True))
            speed_true = torch.sqrt(torch.sum(y_true[..., 1:4]**2, dim=-1, keepdim=True))

            p_ref_pred = torch.mean(y_pred[..., 0:1], dim=1)
            p_ref_true = torch.mean(y_true[..., 0:1], dim=1)

            p_range_pred = torch.max(y_pred[..., 0:1], dim=1)[0] - torch.min(y_pred[..., 0:1], dim=1)[0]
            p_range_true = torch.max(y_true[..., 0:1], dim=1)[0] - torch.min(y_true[..., 0:1], dim=1)[0]

            p_range_true_norm = p_range_true
            speed_range_true = torch.max(speed_true, dim=1)[0] - torch.min(speed_true, dim=1)[0]

            MNAE_pre = torch.mean(torch.abs((y_pred[..., 0:1] - p_ref_pred.unsqueeze(-1)) - (y_true[..., 0:1] - p_ref_true.unsqueeze(-1))) / (p_range_true.unsqueeze(-1) + 1e-8))
            MNAE_speed = torch.mean(torch.abs(speed_pred - speed_true) / (speed_range_true.unsqueeze(-1) + 1e-8))
            MNAE_dp = torch.mean(torch.abs(p_range_pred - p_range_true) / (p_range_true_norm + 1e-8))

            MSE_pre = torch.mean(torch.square((y_pred[..., 0:1] - p_ref_pred.unsqueeze(-1)) - (y_true[..., 0:1] - p_ref_true.unsqueeze(-1))))
            MSE_speed = torch.mean(torch.square(speed_pred - speed_true))
            MSE_dp = torch.mean(torch.square(p_range_pred - p_range_true))

            MAE_pre = torch.mean(torch.abs((y_pred[..., 0:1] - p_ref_pred.unsqueeze(-1)) - (y_true[..., 0:1] - p_ref_true.unsqueeze(-1))))
            MAE_speed = torch.mean(torch.abs(speed_pred - speed_true))
            MAE_dp = torch.mean(torch.abs(p_range_pred - p_range_true))

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

            with open(os.path.join(save_folder, 'inference_results_deeponet.csv'), 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(result_row)

            if save_npy:
                save_x = X[0, :, 0:3].cpu().numpy()
                save_p = (y_pred[0, :, 0:1] - torch.mean(y_pred[0, :, 0:1])).cpu().numpy()
                save_U = y_pred[0, :, 1:4].cpu().numpy()
                save_U[:, 0] = save_U[:, 0] + 0.5

                save_xy = np.concatenate([save_x, save_p, save_U], axis=-1)

                output_file = os.path.join(save_folder, f"deeponet_{case_id}_{flow_rate.replace('=', '')}.npy")
                np.save(output_file, save_xy)

                logger.info(f"Completed inference for case {case_id}, flow rate {flow_rate} (DeepONet): shape={save_xy.shape}, time={inference_time:.4f}s, saved .npy file")
            else:
                logger.info(f"Completed inference for case {case_id}, flow rate {flow_rate} (DeepONet): time={inference_time:.4f}s, no .npy file saved")

    logger.info(f"DeepONet inference completed! Results saved to {save_folder}")

if __name__ == '__main__':
    cfd_path = "/real_data/cfd_data"
    read_folder = "/cfd_opt_deeponet/checkpoint/deeponet"
    save_folder = "/predictions/deeponet"
    os.makedirs(save_folder, exist_ok=True)

    case_id_list = [4]
    test_flow_rates = ['m=0.002', 'm=0.003']

    main(cfd_path, read_folder, save_folder, case_id_range=case_id_list, test_flow_rates=test_flow_rates, use_mixed_precision=False, save_npy=True)
