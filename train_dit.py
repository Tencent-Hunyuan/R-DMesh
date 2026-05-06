import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

import numpy as np
import os
import argparse
import pickle
from tensorboardX import SummaryWriter
from tqdm import tqdm
import json
import shutil

from diffusion.configs import get_model_configs
from networks.configs import model_from_config
from diffusion.rf_diffusion import rf_training_losses_misalign, rf_sample_vc_misalign

def find_latest_ckpt(ckpt_dir):
    ckpt_path = None
    if os.path.exists(ckpt_dir):
        import re
        max_number = -1
        for f in os.listdir(ckpt_dir):
            match = re.match(r'.*_(\d+)\.pth$', f)
            if match:
                number = int(match.group(1))
                if number > max_number:
                    max_number = number
                    ckpt_path = os.path.join(ckpt_dir, f)
    return ckpt_path

def setup_ddp():
    """Initializes the DDP process group using environment variables."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def seed_everything(seed):
    """Sets the random seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

class DyMeshDataset(Dataset):
    def __init__(
            self, 
            data_dir,             # Mesh dir (str 或 list[str])
            latent_data_dir,      # Video Latent dir (str 或 list[str])
            dit_layers=None, 
        ):

        # -------------------------------------------------------
        # 1. Combine data_dir and latent_data_dir
        # -------------------------------------------------------
        if isinstance(data_dir, str): self.data_dirs = [data_dir]
        else: self.data_dirs = data_dir
        if isinstance(latent_data_dir, str): self.latent_data_dirs = [latent_data_dir]
        else: self.latent_data_dirs = latent_data_dir
        assert len(self.data_dirs) == len(self.latent_data_dirs), "Mesh dir should align with Video Latent dir in number!"
        # -------------------------------------------------------
        # 2. Load data
        # -------------------------------------------------------
        self.all_data_items = [] 
        for d_dir, l_dir in zip(self.data_dirs, self.latent_data_dirs):
            mesh_files = set(os.listdir(d_dir))
            latent_files = set(os.listdir(l_dir))
            valid_files_set = mesh_files & latent_files
            valid_files = sorted([f for f in valid_files_set if f.endswith(".bin")])
            for f_name in valid_files:
                item = {
                    'mesh_path': os.path.join(d_dir, f_name),
                    'latent_path': os.path.join(l_dir, f_name),
                    'filename': f_name
                }
                self.all_data_items.append(item)
        self.num_data = len(self.all_data_items)
        print(f"Total data loaded: {self.num_data}")
        if dit_layers is None: self.dit_layers = [10]
        elif isinstance(dit_layers, int): self.dit_layers = [dit_layers]
        else: self.dit_layers = list(dit_layers)
        if 10 not in self.dit_layers:
            raise ValueError("Dit_layers must contain layer 10!")

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        item_data = self.all_data_items[idx % self.num_data]
        mesh_path = item_data['mesh_path']
        base_latent_path = item_data['latent_path']
        # -------------------------------------------------------
        # Load Mesh Latent Data
        # -------------------------------------------------------
        with open(mesh_path, 'rb') as f:
            mesh_file = pickle.load(f)
        x0_latent = torch.tensor(mesh_file['x0_512'], dtype=torch.float32)
        x1_latent = torch.tensor(mesh_file['x1_512'], dtype=torch.float32)
        xt_latent = torch.tensor(mesh_file['xt_512'], dtype=torch.float32)        
        # -------------------------------------------------------
        # Load WAN Video Latents
        # -------------------------------------------------------
        vid_dit_latent_list = []
        for layer_idx in self.dit_layers:
            if layer_idx == 10:
                latent_path = base_latent_path
            else:
                latent_path = base_latent_path.replace("layer_10", f"layer_{layer_idx}")      
            with open(latent_path, 'rb') as f:
                latent_file = pickle.load(f)
            arr = latent_file[f'layer_{layer_idx}']
            t = torch.from_numpy(arr).float() if isinstance(arr, np.ndarray) else torch.tensor(arr, dtype=torch.float32)
            vid_dit_latent_list.append(t)
        return { 
            'x0_latent': x0_latent,
            'x1_latent': x1_latent,
            'xt_latent': xt_latent,
            'vid_dit_latent_list': vid_dit_latent_list,
        }

# @torch.no_grad()
# def validate(vae_model, rf_model, val_loader, vae_factor, device, writer, global_iter, is_ddp, opt):
#     rf_model.eval()
#     x0_mean, x0_std, x1_mean, x1_std, xt_mean, xt_std = vae_factor
#     total_error_sum = torch.tensor(0.0, device=device)     
#     total_valid_count = torch.tensor(0.0, device=device)    
#     for _, data in enumerate(val_loader):
#         if (not is_ddp) or (dist.get_rank() == 0):
#             print("validation iter: ", _)
#         x0_latent = data['x0_latent'].to(device)
#         x1_latent = data['x1_latent'].to(device)
#         xt_latent = data['xt_latent'].to(device)
#         vid_dit_latent = torch.cat(data['vid_dit_latent_list'], dim=-1).to(device)
#         if opt.rescale:
#             x0_start = (x0_latent - x0_mean) / x0_std
#             x1_start = (x1_latent - x1_mean) / x1_std
#             xt_start = (xt_latent - xt_mean) / xt_std
#             x_start = torch.cat([x0_start, x1_start, xt_start], dim=-1)
#         else:
#             x_start = torch.cat([x0_latent, x1_latent, xt_latent], dim=-1)
#         if opt.vid_cond_type == "dit_latent":
#             vid_embed = vid_dit_latent.to(torch.float32)
#         model_kwargs = dict(vid_embed=vid_embed, camera_matrices=None, vid_cond_type=opt.vid_cond_type)
#         x0 = x_start[..., :opt.x0_channels]
#         samples = rf_sample_vc_misalign(
#             model=rf_model, shape=x_start.shape, device=device,
#             model_kwargs=model_kwargs, guidance_scale=1.0,  
#             x0=x0
#         )  
#         if opt.rescale:
#             x0_start_s = samples[..., :opt.x0_channels] * x0_std + x0_mean
#             x1_start_s = samples[..., opt.f0_channels:opt.f0_channels+opt.f1_channels] * x1_std + x1_mean
#             xt_start_s = samples[..., -opt.ft_channels:] * xt_std + xt_mean 
#             samples = torch.cat([x0_start_s, x1_start_s, xt_start_s], dim=-1)
#         outputs = vae_model(vertices, vertices[:, 0], samples=samples, faces=faces, valid_mask=valid_mask, adj_matrix=adj_matrix_nhops, num_traj=opt.num_traj, just_decode=True)
#         # rec error
#         error_rec = outputs - vertices                   
#         euc_dist = torch.norm(error_rec, p=2, dim=-1)               
#         vm = valid_mask.unsqueeze(1)              
#         vm = vm.expand(-1, euc_dist.size(1), -1)        
#         masked_euc_dist = euc_dist * vm
#         total_error_sum += masked_euc_dist.sum()         
#         total_valid_count += vm.sum()                    
#     if is_ddp:
#         dist.all_reduce(total_error_sum, op=dist.ReduceOp.SUM)
#         dist.all_reduce(total_valid_count, op=dist.ReduceOp.SUM)
#     eps = 1e-8
#     avg_euc_dist = (total_error_sum / (total_valid_count + eps)).item()
#     is_main_process = (not is_ddp) or (dist.get_rank() == 0)
#     if is_main_process:
#         print(f'\nValidation at step {global_iter} - Avg Euc Dist over valid points: {avg_euc_dist:.6f}')
#         if writer:
#             writer.add_scalar('val/rec_error', avg_euc_dist, global_iter)
#             writer.flush()
#     rf_model.train()

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Distributed training script for Diffusion model on dynamic meshes.")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name, used for saving checkpoints and logs.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint in the experiment directory.")
    parser.add_argument("--finetune_from", default=None, help="Finetuning from the latest checkpoint in the experiment directory.")
    # Data & Path
    parser.add_argument("--data_dir", type=str, required=True, action="append")
    parser.add_argument("--latent_data_dir", type=str, required=True, action="append")
    parser.add_argument("--dvae_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_dir", type=str, default="./rf_ckpts")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--json_dir", type=str, default="./checkpoints/dvae_factors")
    parser.add_argument("--max_length", type=int, default=4096)
    # VAE
    parser.add_argument("--vae_exp", type=str, default="dvae")
    parser.add_argument("--vae_epoch", type=str, default="2000")
    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size PER GPU.")
    parser.add_argument("--train_epoch", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--rescale", action="store_true")
    # Validation & Saving
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--val_data_dir", default=None)
    parser.add_argument("--val_latent_data_dir", default=None)
    parser.add_argument("--validation_inter", type=int, default=400)
    parser.add_argument("--save_inter", type=int, default=1, help="Save checkpoint every N epochs.")
    # DiT
    parser.add_argument("--base_name", type=str, default="40m", choices=["40m", "300m", "1b"])
    parser.add_argument("--dit_layers", type=int, nargs="+", default=[10])
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    
    opt = parser.parse_args()

    # DDP Setup
    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    local_rank = setup_ddp() if is_ddp else 0
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = not is_ddp or dist.get_rank() == 0

    # Paths and Logging Setup
    exp_dir = os.path.join(opt.save_dir, opt.exp)

    # --- Load VAE and Stats ---
    vae_config_dir = os.path.join(opt.dvae_dir, opt.vae_exp, "model_config.json")
    with open(vae_config_dir, 'r') as f:
        vae_model_config = json.load(f)
        opt.x0_channels = vae_model_config["latent_dim"]
        opt.x1_channels = vae_model_config["latent_dim_x1"]
        opt.xt_channels = vae_model_config["latent_dim"]
        opt.f0_channels = vae_model_config["latent_dim"]
        opt.input_channels = opt.x0_channels + opt.x1_channels + opt.xt_channels
        # if opt.num_t < 0:    
        #     opt.num_t = vae_model_config["T"]
        # else:
        #     vae_model_config["T"] = opt.num_t
        # if opt.num_traj <= 0:
        #     opt.num_traj = vae_model_config["num_traj"]

    # --- Load RF configs ---
    rf_model_config = get_model_configs(opt)
    rf_model_config["dit_layers"] = len(opt.dit_layers)
    
    # all configs
    full_training_config = {
        "vae_config": vae_model_config,
        "rf_config": rf_model_config,
        "training_args": {
            "exp_name": opt.exp,
            "vae_exp_dependency": opt.vae_exp,
            "vae_epoch_dependency": opt.vae_epoch,
            "learning_rate": opt.lr,
            "batch_size_per_gpu": opt.batch_size,
            "total_epochs": opt.train_epoch,
            "seed": opt.seed,
            "rescale_stats": opt.rescale,
        }
    }

    writer = None
    if is_main_process:
        seed_everything(opt.seed)
        print(f"Starting experiment: {opt.exp}")
        os.makedirs(exp_dir, exist_ok=True)
        log_dir = os.path.join(opt.log_dir, opt.exp)
        writer = SummaryWriter(log_dir=str(log_dir), purge_step=None if opt.resume else 0)
        config_save_path = os.path.join(exp_dir, 'training_config.json')
        with open(config_save_path, 'w') as f:
            json.dump(full_training_config, f, indent=4)
        print(f"Full training configuration saved to {config_save_path}")

    if is_ddp:
        dist.barrier() # Ensure all processes have set up paths before proceeding
    
    # vae_model = RDMeshVAE(**vae_model_config).to(device)
    # vae_ckpt_path = os.path.join(opt.dvae_dir, opt.vae_exp, f"dvae_{opt.vae_epoch}.pth")
    # vae_model = load_compatible_checkpoint(vae_model, vae_ckpt_path, device)
    # vae_model.eval()

    # Scale
    if opt.rescale:
        json_path = os.path.join(opt.json_dir, "{}_{}.json".format(opt.vae_exp, opt.vae_epoch))
        if is_main_process:
            dst_json_path = os.path.join(exp_dir, "{}_{}.json".format(opt.vae_exp, opt.vae_epoch))
            shutil.copy(json_path, dst_json_path)
            print(f"Copied VAE stats json from {json_path} to {dst_json_path}")
        with open(json_path, 'r') as f:
            stats = json.load(f)
        x0_mean = torch.tensor(stats['f0_mean'], device=device)
        x0_std = torch.tensor(stats['f0_std'], device=device)
        x1_mean = torch.tensor(stats['f1_mean'], device=device)
        x1_std = torch.tensor(stats['f1_std'], device=device)
        xt_mean = torch.tensor(stats['ft_mean'], device=device)
        xt_std = torch.tensor(stats['ft_std'], device=device)

    # Model, Optimizer
    rf_model = model_from_config(rf_model_config, device)
    optimizer = optim.AdamW(rf_model.parameters(), lr=opt.lr)
    
    # DataLoader Setup 
    dataset = DyMeshDataset(
        opt.data_dir, 
        opt.latent_data_dir,
        dit_layers=opt.dit_layers, 
    )
    sampler = DistributedSampler(dataset) if is_ddp else None
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, sampler=sampler, shuffle=(sampler is None),
                              num_workers=8, pin_memory=True, drop_last=True, prefetch_factor=4, persistent_workers=True)

    # # Val Loader
    # val_loader = None
    # if opt.validate:
    #     val_dataset = DyMeshDataset(
    #         opt.val_data_dir, 
    #         opt.val_latent_data_dir, 
    #         dit_layers=opt.dit_layers, 
    #     )
    #     val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None
    #     val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, sampler=val_sampler,
    #                           num_workers=1, pin_memory=True, drop_last=False)

    # Either resume or finetune 
    if opt.resume and opt.finetune_from is not None:
        raise ValueError("Cannot use --resume and --finetune_from simultaneously.")
        
    # Resume logic
    start_epoch, global_iter = 0, 0
    if opt.resume:
        # ckpt_path = os.path.join(exp_dir, 'latest.pth')
        ckpt_dir = exp_dir
        ckpt_path = find_latest_ckpt(ckpt_dir)
        if ckpt_path is not None and os.path.exists(ckpt_path):
            if is_main_process: print(f"Resuming from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            rf_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            global_iter = checkpoint['global_iter']
        else:
            if is_main_process: print("Resume flag set, but 'latest.pth' not found. Starting from scratch.")
    
    # Finetune logic
    if opt.finetune_from is not None:
        # ckpt_path = os.path.join(opt.save_dir, opt.finetune_from, 'latest.pth')
        ckpt_dir = os.path.join(opt.save_dir, opt.finetune_from)
        ckpt_path = find_latest_ckpt(ckpt_dir)
        if ckpt_path is not None and os.path.exists(ckpt_path):
            if is_main_process: print(f"Finetuning from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            rf_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Finetuning from model: {ckpt_path}")
        else:
            if is_main_process: print("Finetuning flag set, but 'latest.pth' not found. Starting from scratch.")
    
    # DDP Wrapping (must be done after loading weights) 
    if is_ddp:
        rf_model = DistributedDataParallel(rf_model, device_ids=[local_rank], find_unused_parameters=True)

    # Main Training Loop 
    for epoch in range(start_epoch, opt.train_epoch):
        if is_ddp:
            sampler.set_epoch(epoch)
            # if opt.validate:
            #     val_sampler.set_epoch(epoch)
        pbar = dataloader
        if is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{opt.train_epoch}")
        for i, data in enumerate(pbar):
            x0_latent = data['x0_latent'].to(device)
            x1_latent = data['x1_latent'].to(device)
            xt_latent = data['xt_latent'].to(device)
            vid_dit_latent = torch.cat(data['vid_dit_latent_list'], dim=-1).to(device)
            with torch.no_grad():    
                # Normalize          
                if opt.rescale:
                    x0_start = (x0_latent - x0_mean) / x0_std
                    x1_start = (x1_latent - x1_mean) / x1_std
                    xt_start = (xt_latent - xt_mean) / xt_std
                    x_start = torch.cat([x0_start, x1_start, xt_start], dim=-1)
                else:
                    x_start = torch.cat([x0_latent, x1_latent, xt_latent], dim=-1)
                # CFG Random drop
                r = torch.rand(x_start.shape[0], device=device)
                vid_keep_mask = torch.ones(x_start.shape[0], device=device)
                vid_keep_mask[r < opt.cond_drop_prob] = 0 
                vid_embed = vid_dit_latent.to(torch.float32)
                vid_embed = vid_embed * vid_keep_mask[:, None, None]
                model_kwargs = dict(vid_embed=vid_embed, camera_matrices=None, vid_cond_type="dit_latent")
            # Training Step 
            optimizer.zero_grad()
            loss_dict = rf_training_losses_misalign(
                rf_model, x_start, model_kwargs=model_kwargs, 
                x0_channels=opt.x0_channels,
                x1_channels=opt.x1_channels,
                xt_channels=opt.xt_channels,
            )
            loss = loss_dict['loss'].mean()
            loss_x1 = loss_dict['mse_x1'].mean()
            loss_xt = loss_dict['mse_xt'].mean()

            # Skip step on NaN/Inf
            if not torch.isfinite(loss):
                if is_main_process: print(f"Warning: NaN/Inf loss at step {global_iter}. Skipping update.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(rf_model.parameters(), 1.0)
            optimizer.step()
            # lr_scheduler.step()

            # Logging and Validation
            if is_main_process:
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=loss.item(), lr=f"{lr:.2e}")
                writer.add_scalar('train/loss', loss.item(), global_iter)
                writer.add_scalar('train/loss_x1', loss_x1.item(), global_iter)
                writer.add_scalar('train/loss_xt', loss_xt.item(), global_iter)
                writer.add_scalar('train/lr', lr, global_iter)

                
            # if opt.validate and global_iter % opt.validation_inter == 0:
            #     vae_factor = (x0_mean, x0_std, x1_mean, x1_std, xt_mean, xt_std)
            #     if is_main_process:
            #         print("Start Validation!!!")
            #     validate(vae_model, rf_model, val_loader, vae_factor, device, writer, global_iter, is_ddp, opt)
            #     if is_main_process:
            #         print("Validation Ended!!! Continue Training!!!")

            global_iter += 1

        # Save Checkpoint 
        if is_main_process:
            checkpoint = {
                'epoch': epoch,
                'global_iter': global_iter,
                'model_state_dict': rf_model.module.state_dict() if is_ddp else rf_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': lr_scheduler.state_dict(),
            }
            if (epoch + 1) % opt.save_inter == 0:
                torch.save(checkpoint, os.path.join(exp_dir, f'rf_epoch_{epoch+1}.pth'))
            # torch.save(checkpoint, os.path.join(exp_dir, 'latest.pth'))
            print(f"Epoch {epoch+1} finished. Checkpoint saved.")

    # Final Cleanup 
    if is_main_process:
        print("Training finished.")
        writer.close()
    if is_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()

'''




'''