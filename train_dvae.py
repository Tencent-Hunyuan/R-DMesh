import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import numpy as np
import os
from tensorboardX import SummaryWriter
import pickle
import argparse
import json

from networks.rdmeshvae import RDMeshVAE
from utils.mesh_utils import get_adjacency_matrix, calc_n_hops
from utils.loss_functions import loss_function_avg_new, calc_error_avg_new, loss_function_avg_new_per_ins, calc_error_avg_new_per_ins

def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

class DyMeshDataset(Dataset):
    def __init__(self, data_dir, num_t=16, max_length=4096):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".bin")])
        self.num_t = num_t
        self.num_data = len(self.files)
        self.max_length = max_length
        self.faces_max_length = int(self.max_length * 2.5)
    def __len__(self):
        return max(self.num_data, 512)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        with open(file_path, 'rb') as f:
            mesh_file = pickle.load(f)
        vertices, faces = mesh_file["vertices"], mesh_file["faces"]
        vertices, faces = torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int64)
        assert vertices.shape[0] == self.num_t + 1
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
        return vertices, faces, valid_length, valid_mask

class DyMeshDataset_val(Dataset):
    def __init__(self, data_dir, num_t=16, max_length=4096):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".bin")])
        self.num_t = num_t
        self.num_data = len(self.files)
        self.max_length = max_length
        self.faces_max_length = int(self.max_length * 2.5)
    def __len__(self):
        return min(self.num_data, 512) if self.num_data > 0 else 0
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        with open(file_path, 'rb') as f:
            mesh_file = pickle.load(f)
        vertices, faces = mesh_file["vertices"], mesh_file["faces"]
        vertices, faces = torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int64)
        assert vertices.shape[0] == self.num_t + 1
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
        return vertices, faces, valid_length, valid_mask

def train_epoch(model, train_loader, val_loader, optimizer, lr_scheduler, device, epoch, writer, global_iter, opt, is_ddp):
    model.train()
    is_main_process = not is_ddp or dist.get_rank() == 0
    pbar = train_loader
    if is_main_process:
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.train_epoch}", dynamic_ncols=True, initial=global_iter % len(train_loader))
    for b, bc in enumerate(pbar):
        batch, faces, valid_length, valid_mask = bc
        batch, faces, valid_length, valid_mask = batch.to(device), faces.to(device), valid_length.to(device), valid_mask.to(device)
        adj_matrix = get_adjacency_matrix(batch[:, 0], faces, valid_length)
        adj_matrix_nhops = calc_n_hops(adj_matrix, num_hops=opt.num_hops, alpha_hops=opt.alpha_hops, mode=opt.hop_mode, no_norm=True)
        optimizer.zero_grad()
        output = model(batch, batch[:, 0], faces=faces, valid_mask=valid_mask, adj_matrix=adj_matrix_nhops)
        if opt.per_instance_loss:
            loss, loss_recon_opt, loss_recon_real, loss_x1, loss_xt, loss_kl_x1, loss_kl_xt = \
                loss_function_avg_new_per_ins(
                    batch, output, adj_matrix, valid_mask=valid_mask, 
                    alpha_kl=opt.alpha_kl, 
                    sep_rec_loss=opt.sep_rec_loss, 
                    alpha_recon_x1=opt.alpha_recon_x1
                )
        else:
            loss, loss_recon_opt, loss_recon_real, loss_x1, loss_xt, loss_kl_x1, loss_kl_xt = \
                loss_function_avg_new(
                    batch, output, adj_matrix, valid_mask=valid_mask, 
                    alpha_kl=opt.alpha_kl, 
                    sep_rec_loss=opt.sep_rec_loss, 
                    alpha_recon_x1=opt.alpha_recon_x1
                )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # lr_scheduler.step()
        if is_main_process:
            # loss_value = loss.item()
            loss_value = loss_recon_real.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=loss_value, lr=current_lr)
            if writer:
                writer.add_scalar('train/loss', loss.item(), global_iter)
                writer.add_scalar('train/loss_recon_opt', loss_recon_opt.item(), global_iter)
                writer.add_scalar('train/loss_recon_real', loss_recon_real.item(), global_iter)
                writer.add_scalar('train/loss_recon_x1', loss_x1.item(), global_iter)
                writer.add_scalar('train/loss_recon_xt', loss_xt.item(), global_iter)
                writer.add_scalar('train/loss_kl_x1', loss_kl_x1.item(), global_iter)
                writer.add_scalar('train/loss_kl_xt', loss_kl_xt.item(), global_iter)
                writer.add_scalar('train/learning_rate', current_lr, global_iter)
        if opt.validate and global_iter > 0 and global_iter % opt.validation_inter == 0:
            validate(model, val_loader, device, writer, global_iter, is_ddp, opt)
        global_iter += 1
    return global_iter

@torch.no_grad()
def validate(model, val_loader, device, writer, global_iter, is_ddp, opt):
    model.eval()
    local_error_sum = torch.tensor(0.0, device=device)
    local_iters_count = torch.tensor(0, device=device, dtype=torch.long)
    local_weighted_error_sum = torch.tensor(0.0, device=device)  
    local_total_valid_obs = torch.tensor(0, device=device, dtype=torch.long)  
    local_weighted_error_x1_sum = torch.tensor(0.0, device=device)
    local_weighted_error_xt_sum = torch.tensor(0.0, device=device)
    local_total_valid_points = torch.tensor(0, device=device, dtype=torch.long)
    local_strain_error_sum = torch.tensor(0.0, device=device)
    local_abnormal_edges_count = torch.tensor(0, device=device, dtype=torch.long)
    local_total_edges_count = torch.tensor(0, device=device, dtype=torch.long)
    for b, bc in enumerate(val_loader):
        batch, faces, valid_length, valid_mask = bc
        batch, faces, valid_length, valid_mask = batch.to(device), faces.to(device), valid_length.to(device), valid_mask.to(device)
        adj_matrix = get_adjacency_matrix(batch[:, 0], faces, valid_length) 
        adj_matrix_nhops = calc_n_hops(adj_matrix, num_hops=opt.num_hops, alpha_hops=opt.alpha_hops, mode=opt.hop_mode, no_norm=True)
        output = model(batch, batch[:, 0], faces=faces, valid_mask=valid_mask, adj_matrix=adj_matrix_nhops)
        # recon_error, strain_raw_stats = calc_error_avg(batch, output, adj_matrix, valid_mask=valid_mask, alpha_kl=opt.alpha_kl, strain_th=opt.strain_th)
        if opt.per_instance_loss:
            recon_error, rec_error_x1, rec_error_xt, strain_raw_stats = calc_error_avg_new_per_ins(
                batch, output, adj_matrix, valid_mask=valid_mask, strain_th=opt.strain_th
            )
        else:
            recon_error, rec_error_x1, rec_error_xt, strain_raw_stats = calc_error_avg_new(
                batch, output, adj_matrix, valid_mask=valid_mask, strain_th=opt.strain_th
            )
        if valid_mask is not None:
            batch_valid_points = valid_mask.sum() 
            batch_valid_obs = batch_valid_points * batch.shape[1] 
        else:
            batch_valid_points = batch.shape[0] * batch.shape[2]
            batch_valid_obs = batch_valid_points * batch.shape[1]
        local_weighted_error_sum += recon_error * batch_valid_obs
        local_total_valid_obs += batch_valid_obs
        local_weighted_error_x1_sum += rec_error_x1 * batch_valid_points
        local_weighted_error_xt_sum += rec_error_xt * batch_valid_obs 
        local_total_valid_points += batch_valid_points
        local_error_sum += recon_error
        local_iters_count += 1
        local_strain_error_sum += strain_raw_stats['strain_error_sum']
        local_abnormal_edges_count += strain_raw_stats['abnormal_edges_count']
        local_total_edges_count += strain_raw_stats['total_edges_count']
    if is_ddp:
        dist.all_reduce(local_weighted_error_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total_valid_obs, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_weighted_error_x1_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_weighted_error_xt_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total_valid_points, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_error_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_iters_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_strain_error_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_abnormal_edges_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total_edges_count, op=dist.ReduceOp.SUM)
    avg_error_per_point = local_weighted_error_sum / (local_total_valid_obs + 1e-8)
    avg_error_x1 = local_weighted_error_x1_sum / (local_total_valid_points + 1e-8)
    avg_error_xt = local_weighted_error_xt_sum / (local_total_valid_obs + 1e-8)
    epsilon = 1e-8
    global_avg_strain_error = local_strain_error_sum / (local_abnormal_edges_count + epsilon)
    global_abnormal_ratio = local_abnormal_edges_count / (local_total_edges_count + epsilon)
    is_main_process = not is_ddp or dist.get_rank() == 0
    if is_main_process:
        print(f'\nValidation at step {global_iter}')
        print(f'Global Error: {avg_error_per_point.item():.6f} | Shape(X1): {avg_error_x1.item():.6f} | Motion(Xt): {avg_error_xt.item():.6f}')
        print(f'Total Valid Observations: {local_total_valid_obs.item()}')
        # print(f'Average Error per Point (Weighted): {avg_error_per_point.item():.6f}')
        # print(f'Total Valid Observations: {local_total_valid_obs.item()}')
        print(f'Abnormal Edge Ratio: {global_abnormal_ratio.item():.2%} ({local_abnormal_edges_count.item()}/{local_total_edges_count.item()})')
        print(f'Average Strain on Abnormal Edges: {global_avg_strain_error.item():.4f}')
        if writer:
            writer.add_scalar('val/rec_error', avg_error_per_point.item(), global_iter)
            writer.add_scalar('val/rec_error_x1', avg_error_x1.item(), global_iter)
            writer.add_scalar('val/rec_error_xt', avg_error_xt.item(), global_iter)
            writer.add_scalar('val/abnormal_edge_ratio', global_abnormal_ratio.item(), global_iter)
            writer.add_scalar('val/avg_strain_on_abnormal', global_avg_strain_error.item(), global_iter)
            writer.flush()
    model.train()

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpts_dir", type=str, default="./dvae_ckpts")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")
    parser.add_argument("--finetune_from", default=None, help="Finetuning from the latest checkpoint in the experiment directory.")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--val_data_dir", type=str)
    parser.add_argument("--validation_inter", type=int, default=400, help="Validate every N steps.")
    parser.add_argument("--save_inter", type=int, default=1, help="Save every N epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size PER GPU.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial/Max learning rate.")
    parser.add_argument("--train_epoch", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of steps for learning rate warmup.")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--enc_depth", type=int, default=1)
    parser.add_argument("--dec_depth", type=int, default=8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=-1)
    parser.add_argument("--num_t", type=int, default=16)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--latent_dim_x1", type=int, default=8)
    parser.add_argument("--num_traj", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=1)
    ###
    parser.add_argument("--num_hops", type=int, default=1)
    parser.add_argument("--alpha_hops", type=float, default=0.5)
    parser.add_argument("--hop_mode", default="band", choices=["band", "single"])
    parser.add_argument("--alpha_kl", type=float, default=1e-6)
    parser.add_argument("--strain_th", type=float, default=2.0)
    parser.add_argument("--sep_rec_loss", action="store_true")
    parser.add_argument("--alpha_recon_x1", type=float, default=0.1)
    parser.add_argument("--per_instance_loss", action="store_true")
    
    opt = parser.parse_args()

    # DDP Setup
    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    
    local_rank = 0
    if is_ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main_process = not is_ddp or dist.get_rank() == 0
    writer = None
    exp_dir = os.path.join(opt.ckpts_dir, opt.exp)
    
    # DyMeshVAE config
    opt.output_dim = 3 * opt.num_t
    
    model_config = {
        'enc_depth': opt.enc_depth,
        'dec_depth': opt.dec_depth,
        'dim': opt.dim,
        'output_dim': opt.output_dim,  
        'latent_dim': opt.latent_dim,
        'latent_dim_x1': opt.latent_dim_x1,
        'T': opt.num_t,
        'num_traj': opt.num_traj,
        'n_layers': opt.n_layers,
    }

    if is_main_process:
        seed_everything(opt.seed)
        print(f"Running with options: {opt}")
        os.makedirs(exp_dir, exist_ok=True)
        log_dir = os.path.join(opt.log_dir, opt.exp)
        # If resuming, do not overwrite logs
        writer = SummaryWriter(log_dir=str(log_dir), purge_step=None if opt.resume else 0)
        # save config
        config_save_path = os.path.join(exp_dir, 'model_config.json')
        with open(config_save_path, 'w') as f:
            json.dump(model_config, f, indent=4)
        print(f"Model configuration saved to {config_save_path}")

    if is_ddp:
        dist.barrier()
    
    # Dataset and DataLoader setup
    dataset = DyMeshDataset(opt.data_dir, num_t=opt.num_t, max_length=opt.max_length)
    train_sampler = DistributedSampler(dataset) if is_ddp else None
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=8, persistent_workers=True, pin_memory=True, drop_last=True)
    val_loader = None
    if opt.validate and opt.val_data_dir:
        val_dataset = DyMeshDataset_val(opt.val_data_dir, num_t=opt.num_t, max_length=opt.max_length)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, sampler=val_sampler,
                                num_workers=8, persistent_workers=True, pin_memory=True, drop_last=False)
    
    if opt.resume and opt.finetune_from is not None:
        raise ValueError("Cannot use --resume and --finetune_from simultaneously.")
    
    # Model setup
    model = RDMeshVAE(**model_config).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    
    # LR Scheduler setup
    steps_per_epoch = len(train_loader)
    total_training_steps = opt.train_epoch * steps_per_epoch
    lr_scheduler = None
    # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=opt.warmup_steps)
    # main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_training_steps - opt.warmup_steps, eta_min=1e-7)
    # lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[opt.warmup_steps])

    # Start training
    start_epoch = 0
    global_iter = 0
    
    # Process resuming
    if opt.resume:
        # Find the latest checkpoint
        latest_ckpt_path = os.path.join(exp_dir, 'latest.pth')
        if os.path.exists(latest_ckpt_path):
            if is_main_process:
                print(f"Resuming training from checkpoint: {latest_ckpt_path}")
            # Load checkpoint on the correct device
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank} if is_ddp else device
            checkpoint = torch.load(latest_ckpt_path, map_location=map_location, weights_only=False)
            # Load model state
            model_state = checkpoint['model_state_dict']
            if is_ddp and not isinstance(model, DistributedDataParallel):
                 # If current model is not DDP but checkpoint was, strip 'module.' prefix
                model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
            model.load_state_dict(model_state)
            # Load optimizer and scheduler states
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr
            # lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # Load progress
            start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
            global_iter = checkpoint['global_iter']
            if is_main_process:
                print(f"Resumed from epoch {start_epoch}, global step {global_iter}.")
        else:
            if is_main_process:
                print("Resume flag was set, but no 'latest.pth' checkpoint found. Starting from scratch.")
    
    if opt.finetune_from is not None:
        ckpt_path = os.path.join(opt.ckpts_dir, opt.finetune_from, 'latest.pth')
        if os.path.exists(ckpt_path):
            if is_main_process: print(f"Finetuning from checkpoint: {ckpt_path}")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank} if is_ddp else device
            checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Finetuning from model: {ckpt_path}")
        else:
            if is_main_process: print("Finetuning flag set, but 'latest.pth' not found. Starting from scratch.")
    
    # Wrap model with DDP *after* loading state dict
    if is_ddp:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    
    if is_main_process and not opt.resume:
        checkpoint = {
            'epoch': 0,
            'global_iter': global_iter,
            'model_state_dict': model.module.state_dict() if is_ddp else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': lr_scheduler.state_dict(),
            'opt': opt # Optional: save the config as well
        }
        epoch_save_path = os.path.join(exp_dir, f'dvae_0.pth')
        torch.save(checkpoint, epoch_save_path)

    # Training loop
    for epoch in range(start_epoch, opt.train_epoch):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        global_iter = train_epoch(model, train_loader, val_loader, optimizer, lr_scheduler, device, epoch, writer, global_iter, opt, is_ddp)
        if is_main_process:
            # Create a dictionary to save all necessary states
            checkpoint = {
                'epoch': epoch,
                'global_iter': global_iter,
                'model_state_dict': model.module.state_dict() if is_ddp else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': lr_scheduler.state_dict(),
                'opt': opt # Optional: save the config as well
            }
            
            # Save a checkpoint for this specific epoch if save_inter matches
            if (epoch + 1) % opt.save_inter == 0:
                epoch_save_path = os.path.join(exp_dir, f'dvae_{epoch+1}.pth')
                torch.save(checkpoint, epoch_save_path)

            # Always save a 'latest.pth' for easy resuming
            latest_save_path = os.path.join(exp_dir, 'latest.pth')
            torch.save(checkpoint, latest_save_path)
            if (epoch + 1) % opt.save_inter == 0:
                 print(f"Saved checkpoint for epoch {epoch+1} and updated 'latest.pth'")

    # Final save
    if is_main_process:
        # Final model can just be the state dict for inference
        model_to_save = model.module if is_ddp else model
        torch.save(model_to_save.state_dict(), os.path.join(exp_dir, 'dvae_f.pth'))
        if writer:
            writer.close()

    if is_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
    
'''







'''