import torch
import os
import pickle
import json
from networks.rdmeshvae import RDMeshVAE
from utils.mesh_utils import get_adjacency_matrix, merge_identical_vertices, get_edge_lengths_from_verts, calc_n_hops
from utils.render import render_dynamic_mesh_direct_to_video

def main(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    
    # Load data
    test_data_dir = opt.dataset_dir
    ckpt_dir = os.path.join(opt.ckpt_dir, opt.exp, "dvae_{}.pth".format(opt.epoch))
    all_files = sorted(os.listdir(test_data_dir))
    files = all_files[:10]
    dataset_name = opt.dataset_dir.split('/')[-1]
    video_save_dir = os.path.join(opt.video_save_dir, opt.exp+"_{}".format(dataset_name))
    if opt.num_traj > 0:
        video_save_dir = video_save_dir + "_{}".format(opt.num_traj)
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)
    gt_vid_path = video_save_dir+"/epoch_{}/mesh_gt".format(opt.epoch)
    recon_vid_path = video_save_dir+"/epoch_{}/mesh_recon".format(opt.epoch)
    if not os.path.exists(gt_vid_path):
        os.makedirs(gt_vid_path)
    if not os.path.exists(recon_vid_path):
        os.makedirs(recon_vid_path)
    
    # Load model
    config_dir = os.path.join(opt.ckpt_dir, opt.exp, "model_config.json")
    with open(config_dir, 'r') as f:
        model_config = json.load(f)
    model = RDMeshVAE(**model_config).to(device)
    print(f"Loading checkpoint from: {ckpt_dir}")
    checkpoint = torch.load(ckpt_dir, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("  -> Detected new checkpoint format (dictionary).")
        model_weights = checkpoint['model_state_dict']
    else:
        print("  -> Detected old checkpoint format (raw state_dict).")
        model_weights = checkpoint
    model.load_state_dict(model_weights)
    print("Model weights loaded successfully.")
    
    # Metrics
    all_gt_edge_lengths = []
    all_recon_edge_lengths = []
    total_error_sum = 0.0
    total_valid_observations = 0
    
    # Inference
    with torch.no_grad():
        test_count = 0 
        for file in files:
            filepath = os.path.join(test_data_dir, file)
            filename = str(test_count)
            assert file.split('.')[-1] == "bin"
            with open(filepath, 'rb') as f:
                mesh_file = pickle.load(f)
                vertices, faces = mesh_file["vertices"], mesh_file["faces"]
                assert vertices.shape[0] == 65
                vertices, faces = merge_identical_vertices(vertices, faces) 
                vertices, faces = torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int64)
            max_length = max(4096, vertices.shape[1]+512)
            if opt.num_traj < 0:
                opt.num_traj = max(512, max_length // 8)
            if vertices.shape[1] <= opt.min_length:
                print(vertices.shape)
                print("Too few vertices!!!")
                continue
            if vertices.shape[1] > opt.max_length:
                print(vertices.shape)
                print("Too many vertices!!!")
                continue

            # Normalize
            center_start = (vertices[0].max(dim=0)[0] + vertices[0].min(dim=0)[0]) / 2
            center_seq = (vertices[1].max(dim=0)[0] + vertices[1].min(dim=0)[0]) / 2
            vertices_single = vertices[:1] - center_start
            vertices_seq = vertices[1:] - center_seq
            vertices = torch.cat([vertices_single, vertices_seq], dim=0)
            v_max = vertices_single.abs().max()
            v_max = max(v_max, 0.1)
            vertices = vertices / v_max
            faces_max_length = int(opt.max_length * 2.5)
            assert faces.shape[0] <= faces_max_length    
            vertices_ori = vertices
            faces_ori = faces
            
            # Padding
            valid_mask = torch.zeros((1, vertices.shape[1]+10), dtype=torch.bool, device=device)
            valid_mask[:, :vertices.shape[1]] = True
            valid_length = torch.tensor(vertices_ori.shape[1])[None].to(device)
            vertices = torch.cat([vertices, torch.zeros(vertices.shape[0], 10, 3)], dim=1)
            faces = torch.cat([faces, -1 * torch.zeros(10, 3).to(torch.int64)], dim=0)
            pc, query, vertices, faces = vertices[None].to(device), vertices[0][None].to(device), vertices.to(device), faces[None].to(device)
            adj_matrix = get_adjacency_matrix(pc[:, 0], faces, valid_length)
            adj_matrix_nhops = calc_n_hops(adj_matrix, num_hops=opt.num_hops, alpha_hops=opt.alpha_hops, mode=opt.hop_mode, no_norm=True)
            
            # Forward
            output = model(pc, query, faces=faces, valid_mask=valid_mask, adj_matrix=adj_matrix_nhops, num_traj=opt.num_traj)
            recon_pc, pc = output["recon_pc"], output["pc"]
            
            # Average Vertex Error
            error_rec = recon_pc - pc[:, -recon_pc.shape[1]:]
            euc_dist = torch.norm(error_rec, p=2, dim=-1)  # [B, T, V]
            masked_euc_dist = euc_dist * valid_mask.unsqueeze(1)  # [B, T, V]
            total_error_sum += masked_euc_dist.sum().item()
            num_time_steps = pc.shape[1]
            total_valid_observations += valid_mask.sum().item() * num_time_steps

            # Edge length calculation for abnormal edge ratio
            gt_edge_lengths, _, _, _ = get_edge_lengths_from_verts(pc[:, -recon_pc.shape[1]:], adj_matrix, valid_mask)
            recon_edge_lengths, _, _, _ = get_edge_lengths_from_verts(recon_pc, adj_matrix, valid_mask)
            all_gt_edge_lengths.append(gt_edge_lengths.cpu())
            all_recon_edge_lengths.append(recon_edge_lengths.cpu())

            # Render
            faces = faces_ori
            if opt.render:
                print("Start rendering!!!")
                if opt.render_gt:
                    render_dynamic_mesh_direct_to_video(vertices=pc[0].cpu(), face_data=faces, video_save_dir=gt_vid_path, save_name=str(filename), azi=opt.azi, ele=opt.ele)
                render_dynamic_mesh_direct_to_video(vertices=recon_pc[0].cpu(), face_data=faces, video_save_dir=recon_vid_path, save_name=str(filename), azi=opt.azi, ele=opt.ele)
                print("Rendering Ended!!!")
            
            test_count += 1   
            
        # Metrics calculation
        avg_vertex_error = total_error_sum / (total_valid_observations + 1e-8)
        if len(all_gt_edge_lengths) > 0:
            all_gt_edge_l = torch.cat(all_gt_edge_lengths, dim=0).flatten()
            all_recon_edge_l = torch.cat(all_recon_edge_lengths, dim=0).flatten()
            ratio = all_gt_edge_l / (all_recon_edge_l + 1e-9)
            abn2_mask = (ratio > 2.0) | (ratio < 1.0 / 2.0)
            abn5_mask = (ratio > 5.0) | (ratio < 1.0 / 5.0)
            abn10_mask = (ratio > 10.0) | (ratio < 1.0 / 10.0)
            abn2_ratio = abn2_mask.sum().item() / all_gt_edge_l.shape[0]
            abn5_ratio = abn5_mask.sum().item() / all_gt_edge_l.shape[0]
            abn10_ratio = abn10_mask.sum().item() / all_gt_edge_l.shape[0]
        else:
            abn2_ratio = abn5_ratio = abn10_ratio = 0.0
        
        # Print results
        print("="*50)
        print(opt.exp)
        print("Average vertex reconstruction error: ", avg_vertex_error)
        print("Abnormal 2 ratio: ", abn2_ratio)
        print("Abnormal 5 ratio: ", abn5_ratio)
        print("Abnormal 10 ratio: ", abn10_ratio)
        print("="*50)
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--epoch", type=str, default='f')
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--ckpt_dir", type=str, default="./dvae_ckpts")
    parser.add_argument("--video_save_dir", type=str, default="./output_videos")
    parser.add_argument("--min_length", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=50000)
    parser.add_argument("--azi", type=float, default=0.0)
    parser.add_argument("--ele", type=float, default=0.0)
    parser.add_argument("--num_hops", type=int, default=1)
    parser.add_argument("--alpha_hops", type=float, default=0.5)
    parser.add_argument("--hop_mode", default="band", choices=["band", "single"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_gt", action="store_true")
    parser.add_argument("--num_traj", type=int, default=-1)
    
    opt = parser.parse_args()

    main(opt)

'''





'''