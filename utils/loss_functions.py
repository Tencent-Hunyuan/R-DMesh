import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image
import os
from .mesh_utils import get_edge_lengths_from_verts, get_strain_limit_raw_stats

def loss_function_avg_new(
    x, output, adj_matrix, valid_mask=None, 
    alpha_kl=1e-6, 
    sep_rec_loss=False, 
    alpha_recon_x1=1.0 
):
    recon_x, kl_temp = output['recon_pc'], output['kl_temp']
    try:
        kl_x1 = output['kl_x1'] 
    except KeyError:
        kl_x1 = output.get('kl_l1', torch.zeros(1, device=x.device))
    x_target = x[:, -recon_x.shape[1]:] 
    if valid_mask is None:
        valid_mask = torch.ones((x_target.shape[0], x_target.shape[2]), device=x.device, dtype=torch.bool)
    mask_frame = valid_mask 
    mask_seq = valid_mask.unsqueeze(1) 
    # =======================================================
    # Part A: Reconstruction Loss 
    # =======================================================
    # --- 1. Global Reconstruction Loss ---
    dist_global = torch.norm(recon_x - x_target, p=2, dim=-1) # (B, T, N)
    masked_dist_global = dist_global * mask_seq
    loss_rec_global = masked_dist_global.sum() / (mask_seq.sum() * dist_global.shape[1] + 1e-8)
    # --- 2. X1 Reconstruction Loss ---
    recon_x1 = output['recon_x1']  # (B, N, 3)
    gt_x1 = x_target[:, 0]         # (B, N, 3)
    dist_x1 = torch.norm(recon_x1 - gt_x1, p=2, dim=-1) # (B, N)
    masked_dist_x1 = dist_x1 * mask_frame
    loss_x1 = masked_dist_x1.sum() / (mask_frame.sum() + 1e-8)
    # --- 3. Xt Reconstruction Loss ---
    rel_xt = output['rel_xt']      # (B, T, N, 3)
    gt_rel_xt = x_target - x_target[:, :1] # (B, T, N, 3)     
    dist_xt = torch.norm(rel_xt - gt_rel_xt, p=2, dim=-1) # (B, T, N)
    masked_dist_xt = dist_xt * mask_seq
    loss_xt = masked_dist_xt.sum() / (mask_seq.sum() * dist_xt.shape[1] + 1e-8)
    # =======================================================
    # Part B: loss_rec
    # =======================================================
    if sep_rec_loss:
        loss_rec = alpha_recon_x1 * loss_x1 + loss_xt
    else:
        loss_rec = loss_rec_global
    # =======================================================
    # Part C: KL Loss and Final Loss
    # =======================================================
    KLD_temp = kl_temp.mean() 
    KLD_jump = kl_x1.mean() 
    loss = loss_rec + alpha_kl * (KLD_temp + KLD_jump) 
    return loss, loss_rec, loss_rec_global, loss_x1, loss_xt, KLD_jump, KLD_temp

def calc_error_avg_new(x, output, adj_matrix, valid_mask=None, strain_th=2.0):
    recon_x = output['recon_pc']
    x_target = x[:, -recon_x.shape[1]:] 
    if valid_mask is None:
        valid_mask = torch.ones((x_target.shape[0], x_target.shape[2]), device=x.device, dtype=torch.bool)
    mask_frame = valid_mask              # (B, N)
    mask_seq = valid_mask.unsqueeze(1)   # (B, 1, N)
    dist_global = torch.norm(recon_x - x_target, p=2, dim=-1) # (B, T, N)
    rec_error_global = (dist_global * mask_seq).sum() / (mask_seq.sum() * dist_global.shape[1] + 1e-8)
    recon_x1 = output['recon_x1']
    gt_x1 = x_target[:, 0]
    dist_x1 = torch.norm(recon_x1 - gt_x1, p=2, dim=-1) # (B, N)
    rec_error_x1 = (dist_x1 * mask_frame).sum() / (mask_frame.sum() + 1e-8)
    rel_xt = output['rel_xt']
    gt_rel_xt = x_target - x_target[:, :1]
    dist_xt = torch.norm(rel_xt - gt_rel_xt, p=2, dim=-1) # (B, T, N)
    rec_error_xt = (dist_xt * mask_seq).sum() / (mask_seq.sum() * dist_xt.shape[1] + 1e-8)
    gt_edge_lengths, _, _, _ = get_edge_lengths_from_verts(x_target, adj_matrix, valid_mask)
    recon_edge_lengths, _, _, _ = get_edge_lengths_from_verts(recon_x, adj_matrix, valid_mask)
    error_sum, abnormal_count, total_count = get_strain_limit_raw_stats(
        gt_edge_lengths, recon_edge_lengths, threshold=strain_th
    )
    raw_stats = {
        'strain_error_sum': error_sum,
        'abnormal_edges_count': abnormal_count,
        'total_edges_count': total_count
    }
    return rec_error_global, rec_error_x1, rec_error_xt, raw_stats

def loss_function_avg_new_per_ins(
    x, output, adj_matrix, valid_mask=None, 
    alpha_kl=1e-6, 
    sep_rec_loss=False, 
    alpha_recon_x1=1.0 
):
    recon_x, kl_temp = output['recon_pc'], output['kl_temp']
    try:
        kl_x1 = output['kl_x1'] 
    except KeyError:
        kl_x1 = output.get('kl_l1', torch.zeros(1, device=x.device))
    x_target = x[:, -recon_x.shape[1]:] 
    B, T, N, _ = x_target.shape
    if valid_mask is None:
        valid_mask = torch.ones((B, N), device=x.device, dtype=torch.bool)
    mask_frame = valid_mask 
    mask_seq = valid_mask.unsqueeze(1) 
    num_valid_nodes = mask_frame.sum(dim=1).float()
    num_valid_elements = num_valid_nodes * T
    num_valid_nodes = torch.clamp(num_valid_nodes, min=1e-8)
    num_valid_elements = torch.clamp(num_valid_elements, min=1e-8)
    dist_global = torch.norm(recon_x - x_target, p=2, dim=-1) # (B, T, N)
    masked_dist_global = dist_global * mask_seq
    loss_global_per_sample = masked_dist_global.sum(dim=(1, 2)) / num_valid_elements # (B,)
    loss_rec_global = loss_global_per_sample.mean() # Scalar: Batch Average
    recon_x1 = output['recon_x1']  # (B, N, 3)
    gt_x1 = x_target[:, 0]         # (B, N, 3)
    dist_x1 = torch.norm(recon_x1 - gt_x1, p=2, dim=-1) # (B, N)
    masked_dist_x1 = dist_x1 * mask_frame
    loss_x1_per_sample = masked_dist_x1.sum(dim=1) / num_valid_nodes # (B,)
    loss_x1 = loss_x1_per_sample.mean() # Scalar: Batch Average
    rel_xt = output['rel_xt']      # (B, T, N, 3)
    gt_rel_xt = x_target - x_target[:, :1] # (B, T, N, 3)     
    dist_xt = torch.norm(rel_xt - gt_rel_xt, p=2, dim=-1) # (B, T, N)
    masked_dist_xt = dist_xt * mask_seq
    loss_xt_per_sample = masked_dist_xt.sum(dim=(1, 2)) / num_valid_elements # (B,)
    loss_xt = loss_xt_per_sample.mean() # Scalar: Batch Average
    if sep_rec_loss:
        loss_rec = alpha_recon_x1 * loss_x1 + loss_xt
    else:
        loss_rec = loss_rec_global
    KLD_temp = kl_temp.mean() 
    KLD_jump = kl_x1.mean() 
    loss = loss_rec + alpha_kl * (KLD_temp + KLD_jump) 
    return loss, loss_rec, loss_rec_global, loss_x1, loss_xt, KLD_jump, KLD_temp


def calc_error_avg_new_per_ins(x, output, adj_matrix, valid_mask=None, strain_th=2.0):
    recon_x = output['recon_pc']
    x_target = x[:, -recon_x.shape[1]:] 
    B, T, N, _ = x_target.shape
    if valid_mask is None:
        valid_mask = torch.ones((B, N), device=x.device, dtype=torch.bool)
    mask_frame = valid_mask              # (B, N)
    mask_seq = valid_mask.unsqueeze(1)   # (B, 1, N)
    num_valid_nodes = mask_frame.sum(dim=1).float() # (B,)
    num_valid_elements = num_valid_nodes * T        # (B,)
    num_valid_nodes = torch.clamp(num_valid_nodes, min=1e-8)
    num_valid_elements = torch.clamp(num_valid_elements, min=1e-8)
    dist_global = torch.norm(recon_x - x_target, p=2, dim=-1) # (B, T, N)
    rec_error_global = (dist_global * mask_seq).sum(dim=(1, 2)) / num_valid_elements
    rec_error_global = rec_error_global.mean()
    recon_x1 = output['recon_x1']
    gt_x1 = x_target[:, 0]
    dist_x1 = torch.norm(recon_x1 - gt_x1, p=2, dim=-1) # (B, N)
    rec_error_x1 = (dist_x1 * mask_frame).sum(dim=1) / num_valid_nodes
    rec_error_x1 = rec_error_x1.mean()
    rel_xt = output['rel_xt']
    gt_rel_xt = x_target - x_target[:, :1]
    dist_xt = torch.norm(rel_xt - gt_rel_xt, p=2, dim=-1) # (B, T, N)
    rec_error_xt = (dist_xt * mask_seq).sum(dim=(1, 2)) / num_valid_elements
    rec_error_xt = rec_error_xt.mean()
    gt_edge_lengths, _, _, _ = get_edge_lengths_from_verts(x_target, adj_matrix, valid_mask)
    recon_edge_lengths, _, _, _ = get_edge_lengths_from_verts(recon_x, adj_matrix, valid_mask)
    error_sum, abnormal_count, total_count = get_strain_limit_raw_stats(
        gt_edge_lengths, recon_edge_lengths, threshold=strain_th
    )
    raw_stats = {
        'strain_error_sum': error_sum,
        'abnormal_edges_count': abnormal_count,
        'total_edges_count': total_count
    }
    return rec_error_global, rec_error_x1, rec_error_xt, raw_stats

def calculate_psnr_mse(rendered_video_path, reference_video_path, num_frames=64):
    """
    Calculate PSNR and MSE between rendered video and reference video.
    
    Args:
        rendered_video_path (str): Path to rendered video
        reference_video_path (str): Path to reference video
        num_frames (int): Number of frames to compare
    
    Returns:
        dict: Dictionary containing average PSNR and MSE
    """
    # Check if files exist
    if not os.path.exists(rendered_video_path):
        print(f"Warning: Rendered video does not exist: {rendered_video_path}")
        return {'psnr': None, 'mse': None}
    if not os.path.exists(reference_video_path):
        print(f"Warning: Reference video does not exist: {reference_video_path}")
        return {'psnr': None, 'mse': None}
    cap_rendered = cv2.VideoCapture(rendered_video_path)
    cap_reference = cv2.VideoCapture(reference_video_path)
    if not cap_rendered.isOpened():
        print(f"Warning: Cannot open rendered video: {rendered_video_path}")
        return {'psnr': None, 'mse': None}
    if not cap_reference.isOpened():
        print(f"Warning: Cannot open reference video: {reference_video_path}")
        cap_rendered.release()
        return {'psnr': None, 'mse': None}
    psnr_values = []
    mse_values = []
    frame_count = 0
    ref_width = int(cap_reference.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_height = int(cap_reference.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_size = min(ref_width, ref_height)
    for _ in range(num_frames):
        ret_rendered, frame_rendered = cap_rendered.read()
        ret_reference, frame_reference = cap_reference.read()
        if not ret_rendered or not ret_reference:
            break
        img_reference = Image.fromarray(cv2.cvtColor(frame_reference, cv2.COLOR_BGR2RGB))
        img_reference_cropped = TF.center_crop(img_reference, crop_size)
        img_rendered = Image.fromarray(cv2.cvtColor(frame_rendered, cv2.COLOR_BGR2RGB))
        img_rendered_resized = img_rendered.resize((crop_size, crop_size), resample=Resampling.LANCZOS)
        frame_reference_processed = np.array(img_reference_cropped).astype(np.float32) / 255.0
        frame_rendered_processed = np.array(img_rendered_resized).astype(np.float32) / 255.0
        mse = np.mean((frame_reference_processed - frame_rendered_processed) ** 2)
        mse_values.append(mse)
        if mse > 0:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            psnr_values.append(psnr)
        else:
            psnr_values.append(float('inf'))
        frame_count += 1
    cap_rendered.release()
    cap_reference.release()
    if frame_count == 0:
        print("Warning: No frames were successfully compared")
        return {'psnr': None, 'mse': None}
    valid_psnr = [p for p in psnr_values if p != float('inf')]
    avg_psnr = np.mean(valid_psnr) if valid_psnr else float('inf')
    avg_mse = np.mean(mse_values)
    return {
        'psnr': avg_psnr,
        'mse': avg_mse,
        'num_frames_compared': frame_count
    }

