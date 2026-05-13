import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import os
import pickle
import json
import argparse
import numpy as np
from tqdm import tqdm
import pytorch3d.ops as ops

from networks.rdmeshvae import RDMeshVAE
from utils.mesh_utils import get_adjacency_matrix, calc_n_hops

def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        return local_rank, rank, world_size
    else:
        print("DDP environment not detected, running in single GPU mode.")
        return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# -----------------------------
# IO Utils
# -----------------------------
def atomic_write_pickle(obj, path_tmp, path_final):
    """Atomic write to pickle file to avoid corruption during write"""
    with open(path_tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path_tmp, path_final)

def save_latent_to_new_file(output_dir: str, original_bin_name: str, tensor_dict: dict):
    """
    Save latent results to new file with same name in specified directory
    
    Args:
        output_dir: Output directory path
        original_bin_name: Original bin filename (e.g., "example.bin")
        tensor_dict: Dictionary containing latent features to save
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build output file path
    output_path = os.path.join(output_dir, original_bin_name)
    
    # Prepare data for saving
    save_dict = {}
    for k, v in tensor_dict.items():
        if isinstance(v, np.ndarray):
            save_dict[k] = v
        else:
            save_dict[k] = v
    
    # Atomic write
    tmp_path = output_path + ".tmp"
    atomic_write_pickle(save_dict, tmp_path, output_path)
    
    return output_path

class DyMeshDataset(Dataset):
    def __init__(self, data_dir, num_t=65, max_length=4096):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".bin")])    
        self.num_t = num_t + 1
        self.max_length = max_length
        self.faces_max_length = int(self.max_length * 2.5) 
        assert self.num_t % 4 == 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        file_path = os.path.join(self.data_dir, filename)

        with open(file_path, 'rb') as f:
            mesh_file = pickle.load(f)
        
        vertices, faces = mesh_file["vertices"], mesh_file["faces"]
        vertices = torch.tensor(vertices, dtype=torch.float32)
        faces = torch.tensor(faces, dtype=torch.int64)

        assert vertices.shape[0] >= self.num_t
        vertices = vertices[:self.num_t]

        # Center & Scale
        # center_start = (vertices[0].max(dim=0)[0] + vertices[0].min(dim=0)[0]) / 2
        # center_seq = (vertices[1].max(dim=0)[0] + vertices[1].min(dim=0)[0]) / 2
        center_start = vertices[0].mean(dim=0)
        center_seq = vertices[1].mean(dim=0)
        vertices_cond = vertices[:1] - center_start
        vertices_seq = vertices[1:] - center_seq
        vertices = torch.cat([vertices_cond, vertices_seq], dim=0)
        v_max = vertices_cond.abs().max()
        v_max = max(v_max, 0.1)
        vertices = vertices / v_max

        current_t = vertices.shape[0]
        valid_length = vertices.shape[1] # 当前 Mesh 的顶点数

        # Padding Vertices
        valid_mask = torch.zeros(self.max_length, dtype=torch.bool)
        valid_mask[:valid_length] = True
        padded_vertices = torch.zeros(current_t, self.max_length, 3, dtype=torch.float32)
        padded_vertices[:, :valid_length] = vertices

        # Padding Faces
        padded_faces = -torch.ones(self.faces_max_length, 3, dtype=torch.int64)
        num_faces = faces.shape[0]
        if num_faces > self.faces_max_length:
            faces = faces[:self.faces_max_length]
            num_faces = self.faces_max_length
        padded_faces[:num_faces] = faces

        return {
            'vertices': padded_vertices, 
            'faces': padded_faces, 
            'valid_length': valid_length, 
            'valid_mask': valid_mask,
            'file_path': file_path,
            'bin_name': filename  # 添加文件名用于输出
        }

def main(opt):
    local_rank, global_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')

    # Set seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    
    # --- Load Model Config & Weights ---
    ckpt_path = os.path.join(opt.ckpt_dir, opt.exp, "dvae_{}.pth".format(opt.epoch))
    config_path = os.path.join(opt.ckpt_dir, opt.exp, "model_config.json")
    
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    if global_rank == 0:
        print(f"Loading model from {ckpt_path}...")
    
    model_config["T"] = opt.num_t

    model = RDMeshVAE(**model_config).to(device)

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("  -> Detected new checkpoint format (dictionary).")
        model_weights = checkpoint['model_state_dict']
    else:
        print("  -> Detected old checkpoint format (raw state_dict).")
        model_weights = checkpoint
    model.load_state_dict(model_weights, strict=False)
    print("Model weights loaded successfully.")
    
    model.eval()
    
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # --- Dataset & Dataloader ---
    dataset = DyMeshDataset(
        opt.dataset_dir, 
        num_t=opt.num_t, 
        max_length=opt.max_length,
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=False) if dist.is_initialized() else None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=False
    )

    if global_rank == 0:
        print(f"Start processing {len(dataset)} files with {world_size} GPUs. Batch size: {opt.batch_size}")

    # --- Inference Loop ---
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Extracting", unit="batch") if global_rank == 0 else dataloader
        
        for batch in iterator:
            # Move data to GPU
            # vertices: [B, T, max_len, 3]
            vertices = batch['vertices'].to(device)
            # faces: [B, max_faces, 3]
            faces = batch['faces'].to(device)
            # valid_length: [B] 
            valid_length = batch['valid_length'].to(device)
            # valid_mask: [B, max_len]
            valid_mask = batch['valid_mask'].to(device)
            file_paths = batch['file_path']
            bin_names = batch['bin_name']

            # Adjacency Matrix
            adj_matrix = get_adjacency_matrix(vertices[:, 0], faces, valid_length)
            
            adj_matrix_nhops = calc_n_hops(adj_matrix, num_hops=opt.num_hops, alpha_hops=opt.alpha_hops, mode=opt.hop_mode, no_norm=True)

            # --- Forward Pass (Encode Only) ---
            query = vertices[:, 0] 
            x = model(vertices, query, faces=faces, valid_mask=valid_mask, 
                           adj_matrix=adj_matrix_nhops, num_traj=opt.num_traj, 
                           just_encode=True)
            
            x0 = x[..., :model_config["latent_dim"]]
            x1 = x[..., model_config["latent_dim"]:model_config["latent_dim"]+model_config["latent_dim_x1"]]
            xt = x[..., -model_config["latent_dim"]:]
           

            # --- Save Results ---
            x0_np = x0.cpu().numpy() # [B, K, C]
            x1_np = x1.cpu().numpy() # [B, K, C]
            xt_np = xt.cpu().numpy() # [B*T, K, C]

            for b in range(len(file_paths)):
                bin_name = bin_names[b]
                current_x0 = x0_np[b]
                current_x1 = x1_np[b]
                current_xt = xt_np[b] # [T, K, C]
                
                latent_dict = {
                    'x0_{}'.format(opt.num_traj): current_x0,
                    'x1_{}'.format(opt.num_traj): current_x1,
                    'xt_{}'.format(opt.num_traj): current_xt
                }

                try:
                    output_path = save_latent_to_new_file(opt.output_dir, bin_name, latent_dict)
                    if global_rank == 0:
                        print(f"Saved latent features to: {output_path}")
                except Exception as e:
                    print(f"Error saving {bin_name}: {e}")

    cleanup_ddp()
    if global_rank == 0:
        print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--ckpt_dir", type=str, default="./dvae_ckpts")
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--epoch", type=str, default='f')
    
    # Data Params
    parser.add_argument("--max_length", type=int, default=4096, help="Max vertices for padding")
    parser.add_argument("--num_t", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    
    # Model Params
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--num_hops", type=int, default=1)
    parser.add_argument("--alpha_hops", type=float, default=0.5)
    parser.add_argument("--hop_mode", default="band", choices=["band", "single"])
    parser.add_argument("--flext", action="store_true")
    parser.add_argument("--no_ptla", action="store_true")
    parser.add_argument("--no_vnor", action="store_true")
    parser.add_argument("--no_tdgw", action="store_true")
    parser.add_argument("--infer_81", action="store_true")
    parser.add_argument("--valid_diag", action="store_true")
    
    # Extraction Params
    parser.add_argument("--num_traj", type=int, required=True, help="Target number of tokens (K)")
    
    # Output Params
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for latent features")

    opt = parser.parse_args()
    main(opt)

'''





'''
