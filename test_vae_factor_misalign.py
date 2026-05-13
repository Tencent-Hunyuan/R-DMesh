### test_vae_factor_misalign.py: 每个通道单独计算mean/std

import os
import json
import time
import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist

from tqdm import tqdm
from networks.dymeshvae_tri import DyMeshVAE_TriFlow
from utils.mesh_utils import get_adjacency_matrix, calc_n_hops


def setup_ddp():
    """Initializes the DDP process group using environment variables."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def load_compatible_checkpoint(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_weights = checkpoint['model_state_dict']
    else:
        model_weights = checkpoint
    if list(model_weights.keys())[0].startswith('module.'):
        model_weights = {k.replace('module.', ''): v for k, v in model_weights.items()}
    model.load_state_dict(model_weights)
    print("Model weights loaded successfully.")
    return model


class OnlineStats:
    """
    Scalar online aggregator using sum, sumsq, and count.
    """
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.n = torch.tensor(0, dtype=torch.long, device=self.device)
        self.sum = torch.tensor(0.0, dtype=torch.double, device=self.device)
        self.sumsq = torch.tensor(0.0, dtype=torch.double, device=self.device)

    def update(self, batch: torch.Tensor):
        x = batch.to(self.device, dtype=torch.double)
        cnt = x.numel()
        if cnt == 0:
            return
        self.n += cnt
        self.sum += x.sum()
        self.sumsq += (x * x).sum()

    @property
    def avg(self):
        n = self.n.item()
        if n == 0:
            return 0.0
        return (self.sum / self.n).item()

    @property
    def std(self):
        n = self.n.item()
        if n < 2:
            return 0.0
        mean = self.sum / self.n
        var = (self.sumsq - self.n * mean * mean) / (self.n - 1)
        var = torch.clamp(var, min=0.0)
        return torch.sqrt(var).item()

    def get_state(self):
        return torch.stack([self.n.double(), self.sum, self.sumsq])

    def set_state(self, state_tensor: torch.Tensor):
        self.n = state_tensor[0].long()
        self.sum = state_tensor[1]
        self.sumsq = state_tensor[2]


class OnlineStatsPerChannel:
    """
    Per-channel online aggregator using vectorized sum/sumsq and shared count.
    Assumes the input has a channel dimension C on the last axis.
    """
    def __init__(self, num_channels: int, device='cpu'):
        self.device = torch.device(device)
        self.C = int(num_channels)
        self.n = torch.tensor(0, dtype=torch.long, device=self.device)  # total elements per channel (shared count)
        self.sum = torch.zeros(self.C, dtype=torch.double, device=self.device)
        self.sumsq = torch.zeros(self.C, dtype=torch.double, device=self.device)

    def update(self, batch: torch.Tensor):
        """
        batch: any shape [..., C] where C == self.C.
        We reduce over all non-channel dims.
        """
        x = batch.to(self.device, dtype=torch.double)
        assert x.shape[-1] == self.C, f"Expected last dim {self.C}, got {x.shape[-1]}"
        # elements per channel in this batch = prod(x.shape[:-1])
        cnt = int(np.prod(x.shape[:-1]))  # uses CPU integer; safe
        if cnt == 0:
            return
        # reshape to [-1, C], sum over rows
        x2d = x.reshape(-1, self.C)
        self.n += cnt
        self.sum += x2d.sum(dim=0)
        self.sumsq += (x2d * x2d).sum(dim=0)

    def mean(self) -> torch.Tensor:
        n = int(self.n.item())
        if n == 0:
            return torch.zeros_like(self.sum)
        return self.sum / self.n

    def std(self) -> torch.Tensor:
        n = int(self.n.item())
        if n < 2:
            return torch.zeros_like(self.sum)
        m = self.sum / self.n
        var = (self.sumsq - self.n * m * m) / (self.n - 1)
        var = torch.clamp(var, min=0.0)
        return torch.sqrt(var)

    def get_state(self) -> torch.Tensor:
        """
        Pack as a single 1D tensor for all_reduce:
        [n, sum(0..C-1), sumsq(0..C-1)] with doubles.
        """
        return torch.cat([
            self.n.view(1).double(),
            self.sum.double(),
            self.sumsq.double()
        ], dim=0)

    def set_state(self, state_tensor: torch.Tensor):
        """
        Unpack after all_reduce SUM.
        """
        assert state_tensor.numel() == 1 + 2 * self.C
        self.n = state_tensor[0].long()
        self.sum = state_tensor[1:1 + self.C]
        self.sumsq = state_tensor[1 + self.C:1 + 2 * self.C]


class DyMeshDataset(Dataset):
    def __init__(self, data_dir, max_length=4096):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".bin")])[:50000]
        self.num_data = len(self.files)
        self.max_length = max_length
        self.faces_max_length = int(self.max_length * 2.5)
    def __len__(self):
        return self.num_data
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        with open(file_path, 'rb') as f:
            mesh_file = pickle.load(f)
        vertices, faces = mesh_file["vertices"], mesh_file["faces"]
        vertices, faces = torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int64)
        frame_cond = vertices[0:1] 
        frame_seq = vertices[1:]
        # center_cond = (frame_cond[0].max(dim=0)[0] + frame_cond[0].min(dim=0)[0]) / 2
        center_cond = frame_cond[0].mean(dim=0)
        frame_cond = frame_cond - center_cond
        # center_seq = (frame_seq[0].max(dim=0)[0] + frame_seq[0].min(dim=0)[0]) / 2
        center_seq = frame_seq[0].mean(dim=0)
        frame_seq = frame_seq - center_seq
        v_max = max(0.1, frame_cond.abs().max() + 1e-8)
        frame_cond = frame_cond / v_max
        frame_seq = frame_seq / v_max
        vertices = torch.cat([frame_cond, frame_seq], dim=0)
        valid_length = vertices.shape[1]
        valid_mask = torch.cat([torch.ones(valid_length, dtype=torch.bool), torch.zeros((self.max_length-valid_length), dtype=torch.bool)], dim=0)
        vertices = torch.cat([vertices, torch.zeros(vertices.shape[0], self.max_length-vertices.shape[1], 3)], dim=1)
        faces = torch.cat([faces, -1 * torch.ones(self.faces_max_length-faces.shape[0], 3).to(torch.int64)], dim=0)
        return {'vertices': vertices, 'faces': faces, 'valid_length': valid_length, 'valid_mask': valid_mask}

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def round_list(x, ndigits=4):
    # x: list[float]
    return [round(float(v), ndigits) for v in x]

def main():
    parser = argparse.ArgumentParser(description="Compute per-channel latent statistics for a dynamic mesh dataset.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--vae_dir", type=str, default="./dvae_ckpts")
    parser.add_argument("--save_dir", type=str, default="./dvae_factors")
    parser.add_argument("--vae_exp", type=str, required=True)
    parser.add_argument("--vae_epoch", type=str, default='1000')
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_hops", type=int, default=4)
    parser.add_argument("--alpha_hops", type=float, default=0.5)
    parser.add_argument("--hop_mode", default="band", choices=["band", "single"])
    parser.add_argument("--flext", action="store_true")
    parser.add_argument("--num_t", type=int, default=-1)
    opt = parser.parse_args()

    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    local_rank = setup_ddp() if is_ddp else 0
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = not is_ddp or dist.get_rank() == 0

    if is_main_process:
        seed_everything(opt.seed)
        print("Starting feature statistics calculation (per-channel)...")
        print(f"Running in {'DDP' if is_ddp else 'single-GPU'} mode.")

    # DataLoader
    dataset = DyMeshDataset(opt.data_dir, max_length=opt.max_length)
    sampler = DistributedSampler(dataset, shuffle=False) if is_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    # VAE Model Loading
    config_dir = os.path.join(opt.vae_dir, opt.vae_exp, "model_config.json")
    with open(config_dir, 'r') as f:
        model_config = json.load(f)
    opt.latent_dim = model_config["latent_dim"]
    opt.latent_dim_x1 = model_config["latent_dim_x1"]
    opt.dim = model_config["dim"]
    if opt.num_t > 0:
        model_config["T"] = opt.num_t
    vae_model = DyMeshVAE_TriFlow(**model_config).to(device)
    vae_ckpt = os.path.join(opt.vae_dir, opt.vae_exp, f"dvae_{opt.vae_epoch}.pth")
    vae_model = load_compatible_checkpoint(vae_model, vae_ckpt, device)
    vae_model.eval()

    # Per-channel online stats for f0 and ft
    f0_stats = OnlineStatsPerChannel(num_channels=opt.latent_dim, device=device)
    f1_stats = OnlineStatsPerChannel(num_channels=opt.latent_dim_x1, device=device)
    ft_stats = OnlineStatsPerChannel(num_channels=opt.latent_dim, device=device)

    # Progress bar only on main process
    pbar = dataloader if not is_main_process else tqdm(dataloader, desc="Processing batches")

    with torch.no_grad():
        for data in pbar:
            x = data['vertices'].to(device)
            faces = data['faces'].to(device)
            valid_length = data['valid_length'].to(device)

            adj_matrix = get_adjacency_matrix(x[:, 0], faces, valid_length)
            try:
                adj_matrix_nhops = calc_n_hops(
                    adj_matrix,
                    num_hops=opt.num_hops,
                    alpha_hops=opt.alpha_hops,
                    mode=opt.hop_mode
                )
            except Exception as e:
                if is_main_process:
                    print(f"calc_n_hops failed; fallback to 1-hop. Err: {e}")
                adj_matrix_nhops = adj_matrix
            
           
            x_latent = vae_model(
                x,
                queries=x[:, 0],
                faces=faces,
                valid_mask=data['valid_mask'].to(device),
                adj_matrix=adj_matrix_nhops,
                just_encode=True
            )

            # Split into f0, ft on channel-last dim
            f0_batch = x_latent[..., :opt.latent_dim]
            f1_batch = x_latent[..., opt.latent_dim:opt.latent_dim+opt.latent_dim_x1]
            ft_batch = x_latent[..., -opt.latent_dim:]

            # Update per-channel stats (reduce over all non-channel dims)
            f0_stats.update(f0_batch)  # shape [..., C]
            f1_stats.update(f1_batch)  # shape [..., C]
            ft_stats.update(ft_batch)  # shape [..., C]

    # DDP aggregate: all_reduce SUM on packed states
    if is_ddp:
        if is_main_process:
            print("Aggregating statistics across all processes...")

        f0_state = f0_stats.get_state()
        f1_state = f1_stats.get_state()
        ft_state = ft_stats.get_state()

        dist.all_reduce(f0_state, op=dist.ReduceOp.SUM)
        dist.all_reduce(f1_state, op=dist.ReduceOp.SUM)
        dist.all_reduce(ft_state, op=dist.ReduceOp.SUM)

        if is_main_process:
            f0_stats.set_state(f0_state)
            f1_stats.set_state(f1_state)
            ft_stats.set_state(ft_state)

    # Save results (only main process)
    if is_main_process:
        f0_mean = f0_stats.mean().cpu().tolist()
        f0_std = f0_stats.std().cpu().tolist()
        f1_mean = f1_stats.mean().cpu().tolist()
        f1_std = f1_stats.std().cpu().tolist()
        ft_mean = ft_stats.mean().cpu().tolist()
        ft_std = ft_stats.std().cpu().tolist()

        f0_mean = round_list(f0_mean, 4)
        f0_std  = round_list(f0_std, 4)
        f1_mean = round_list(f1_mean, 4)
        f1_std  = round_list(f1_std, 4)
        ft_mean = round_list(ft_mean, 4)
        ft_std  = round_list(ft_std, 4)

        stats_dict = {
            'f0_mean': f0_mean,   # list of length latent_dim
            'f0_std': f0_std,     # list of length latent_dim
            'f1_mean': f1_mean,   # list of length latent_dim_x1
            'f1_std': f1_std,     # list of length latent_dim_x1
            'ft_mean': ft_mean,   # list of length latent_dim
            'ft_std': ft_std      # list of length latent_dim
        }

        os.makedirs(opt.save_dir, exist_ok=True)
        save_path = os.path.join(opt.save_dir, f"{opt.vae_exp}_{opt.vae_epoch}.json")
        with open(save_path, 'w') as f:
            json.dump(stats_dict, f, indent=4)

        print(f"\nPer-channel statistics saved to {save_path}")
        print(f"Example: f0_mean[0]={f0_mean[0]:.6f}, f0_std[0]={f0_std[0]:.6f}")
        print(f"Example: f1_mean[0]={f1_mean[0]:.6f}, f1_std[0]={f1_std[0]:.6f}")
        print(f"Example: ft_mean[0]={ft_mean[0]:.6f}, ft_std[0]={ft_std[0]:.6f}")


if __name__ == '__main__':
    main()

'''

torchrun --nproc_per_node=8 \
test_vae_factor_misalign.py \
--data_dir /mnt/zw2/zijiewu/Datasets/objxl_animation_bins_merged_4096_65f_filtered_trainset \
--max_length 4096 --batch_size 16 \
--vae_dir /mnt/zw/zijiewu/checkpoints \
--vae_exp dvae_nhops4_n2_new_enc8_64f_misalign_center_triflow_sep_recon_1218 \
--num_hops 4 --vae_epoch 150 --num_t 64

torchrun --nproc_per_node=8 \
test_vae_factor_misalign.py \
--data_dir /mnt/zw2/zijiewu/Datasets/objxl_animation_bins_merged_4096_65f_filtered_trainset \
--max_length 4096 --batch_size 16 \
--vae_dir /mnt/zw/zijiewu/checkpoints \
--vae_exp dvae_nhops4_n2_new_enc8_64f_misalign_center_triflow_sep_recon_per_ins_1218 \
--num_hops 4 --vae_epoch 150 --num_t 64

torchrun --nproc_per_node=8 \
test_vae_factor_misalign.py \
--data_dir /mnt/zw2/zijiewu/Datasets/objxl_animation_bins_merged_4096_65f_filtered_trainset \
--max_length 4096 --batch_size 16 \
--vae_dir /mnt/zw/zijiewu/checkpoints \
--vae_exp dvae_nhops4_n2_new_enc8_64f_misalign_center_triflow_sep_recon_per_ins_1218 \
--num_hops 4 --vae_epoch 190 --num_t 64

torchrun --nproc_per_node=8 \
test_vae_factor_misalign.py \
--data_dir /mnt/zw2/zijiewu/Datasets/objxl_animation_bins_merged_4096_65f_filtered_trainset \
--max_length 4096 --batch_size 16 \
--vae_dir /mnt/zw/zijiewu/checkpoints \
--vae_exp dvae_nhops4_n2_new_enc8_64f_misalign_center_triflow_sep_recon_per_ins_1218 \
--num_hops 4 --vae_epoch 330 --num_t 64

'''