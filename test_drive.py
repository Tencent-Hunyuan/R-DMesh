import torch
from torch.utils.data import Dataset, DataLoader
import os
import json

from diffusion.rf_diffusion import rf_sample_vc_misalign
from networks.configs import model_from_config
from networks.rdmeshvae import RDMeshVAE
from utils.mesh_utils import get_adjacency_matrix, merge_identical_vertices_with_indices, calc_n_hops
from utils.render import full_blender_cleanup, get_all_vertices, get_all_faces, import_model
from utils.render_texture import drive_mesh_and_render_with_pkl, drive_mesh_and_render_with_pkl_frames
from utils.data_utils import load_png_to_tensor, load_mp4_to_tensor

import Wan2_2.wan as wan
from Wan2_2.wan.configs import WAN_CONFIGS

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
    model.load_state_dict(model_weights, strict=False)
    print("Model weights loaded successfully.")
    return model
 
class DyMeshDataset(Dataset):
    def __init__(self, mesh_dir, video_dir, num_t=64, video_width=256, num_test=10, mesh_list=None, video_list=None):
        self.mesh_dir = mesh_dir
        self.video_dir = video_dir
        all_meshes = (
            mesh_list
            if mesh_list is not None
            else sorted(f for f in os.listdir(mesh_dir) if f.endswith('.fbx') or f.endswith('.glb'))
        )
        all_videos = (
            video_list
            if video_list is not None
            else sorted(v for v in os.listdir(video_dir) if v.endswith('.mp4'))
        )
        self.meshes = []
        self.videos = []
        for f in all_meshes:
            for v in all_videos:
                self.meshes.append(f)
                self.videos.append(v)     
        self.num_t = num_t
        self.num_data = min(num_test, len(self.videos))
        self.video_width = video_width

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # mesh
        mesh_path = os.path.join(self.mesh_dir, self.meshes[idx])
        # video
        video_path = os.path.join(self.video_dir, self.videos[idx])
        try:
            video_tensor = load_mp4_to_tensor(video_path, num_frames=self.num_t, video_width=self.video_width)
        except:
            video_tensor = load_png_to_tensor(video_path, num_frames=self.num_t, video_width=self.video_width)
        video_tensor = torch.cat([video_tensor[:, :1], video_tensor], dim=1)
        assert video_tensor.shape[1] == self.num_t + 1
        seq_len = (self.num_t // 4 + 1) * self.video_width * self.video_width // 1024
        video_name = self.meshes[idx].split('.')[0]+"_"+self.videos[idx].split('.')[0]
        return {
            'mesh_path': mesh_path, 
            'video_tensor': video_tensor,
            'seq_len': seq_len,
            'video_name': video_name
        }


def main(opt):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set seed
    seed = opt.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Video save dir
    if opt.testset is not None:
        video_save_dir = os.path.join(opt.video_save_dir, opt.rf_exp, opt.rf_epoch, opt.testset)
    else:
        video_save_dir = os.path.join(opt.video_save_dir, opt.rf_exp, opt.rf_epoch)
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)
    
    # Load the unified training configuration file
    print("Loading unified training configuration...")
    config_path = os.path.join(opt.rf_model_dir, opt.rf_exp, "training_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Unified config not found at: {config_path}. This file is essential for inference.")
    
    with open(config_path, 'r') as f:
        full_config = json.load(f)

    # Extract the individual configurations
    vae_config = full_config['vae_config']
    rf_config = full_config['rf_config']
    training_args = full_config['training_args']
    opt.x0_channels = vae_config["latent_dim"]
    opt.x1_channels = vae_config["latent_dim_x1"]
    opt.xt_channels = vae_config["latent_dim"]
    opt.f0_channels = vae_config["latent_dim"]
    if opt.num_t < 0:    
        opt.num_t = vae_config["T"]
    else:
        vae_config["T"] = opt.num_t
    opt.vae_exp = training_args["vae_exp_dependency"]
    opt.vae_epoch = training_args["vae_epoch_dependency"]
    print("Configuration loaded successfully.")
    
    # Load rescale params
    if opt.rescale:
        json_path = os.path.join(opt.json_dir, "{}_{}.json".format(opt.vae_exp, opt.vae_epoch))
        with open(json_path, 'r') as f:
            stats = json.load(f)
        x0_mean = torch.tensor(stats['f0_mean'], device=device)
        x0_std = torch.tensor(stats['f0_std'], device=device)
        x1_mean = torch.tensor(stats['f1_mean'], device=device)
        x1_std = torch.tensor(stats['f1_std'], device=device)
        xt_mean = torch.tensor(stats['ft_mean'], device=device)
        xt_std = torch.tensor(stats['ft_std'], device=device)

    # RDMeshVAE model
    print("Loading RDMeshVAE...")
    vae_dir = os.path.join(opt.vae_dir, opt.vae_exp, "dvae_{}.pth".format(opt.vae_epoch))
    vae_model = RDMeshVAE(**vae_config).to(device)
    vae_model = load_compatible_checkpoint(vae_model, vae_dir, device)
    vae_model.eval()
    print("RDMeshVAE loaded!!!")
    
    # RDMeshDiT model
    print("Loading RDMeshDiT Model...")
    rf_model_dir = os.path.join(opt.rf_model_dir, opt.rf_exp, "rf_epoch_{}.pth".format(opt.rf_epoch))
    rf_model = model_from_config(rf_config, device)
    rf_model = load_compatible_checkpoint(rf_model, rf_model_dir, device)
    rf_model.eval()
    print("RDMeshDiT Model loaded!!!")
    
    # Wan model
    print("Loading Wan2.2-TI2V-5B...")
    cfg = WAN_CONFIGS["ti2v-5B"]
    wan_ti2v = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=opt.wan_model_dir,
        convert_model_dtype=True,
    )
    print("Wan loaded!!!")

    # Test dataset
    dataset = DyMeshDataset(
        opt.data_dir, 
        opt.video_data_dir, 
        num_t=opt.num_t, 
        video_width=opt.video_width, 
        num_test=opt.num_test,
        mesh_list=opt.mesh_list,
        video_list=opt.video_list
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
 
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            mesh_path = data['mesh_path'][0]
            video_tensor = data['video_tensor'].to(device)
            seq_len = data['seq_len'][0]
            video_name = data["video_name"][0]
            
            # Extract WAN latents
            video_latents = wan_ti2v.v2l_sample(videos=video_tensor, seq_len=seq_len, samp_layers=opt.dit_layers, samp_ratio=opt.samp_ratio, ori_samp=opt.ori_samp)
            for l in range(len(opt.dit_layers)):
                dit_tensor = torch.tensor(video_latents[f"layer_{opt.dit_layers[l]}"], dtype=torch.float32, device=device)
                if l == 0:
                    vid_dit_latent = dit_tensor
                else:
                    vid_dit_latent = torch.cat([vid_dit_latent, dit_tensor], dim=-1)
            vid_embed = vid_dit_latent
            
            # Import the mesh file
            full_blender_cleanup()
            mesh_objects = import_model(mesh_path, frame_idx=0)
            all_vertices = get_all_vertices(mesh_objects)
            all_faces = get_all_faces(mesh_objects)
            merged_verts, merged_faces, all_indices = merge_identical_vertices_with_indices(all_vertices, all_faces)
            vertices, faces = torch.tensor(merged_verts, dtype=torch.float32), torch.tensor(merged_faces, dtype=torch.int64)
            
            # Data pre-processing
            center = vertices.mean(dim=0)
            vertices = vertices - center
            v_max = max(0.1, vertices.abs().max())
            vertices = vertices / (v_max + 1e-8)
            vertices = vertices[None, None].to(device)
            vertices = vertices.repeat(1, opt.num_t+1, 1, 1) 
            faces = faces[None].to(device)
            valid_mask = ~(vertices.permute(0, 2, 1, 3).flatten(2, 3) == 0.0).all(dim=-1)
            valid_length = valid_mask.sum(dim=-1)
            adj_matrix = get_adjacency_matrix(vertices[:, 0], faces, valid_length)
            adj_matrix_nhops = calc_n_hops(adj_matrix, num_hops=opt.num_hops, alpha_hops=opt.alpha_hops, mode=opt.hop_mode, no_norm=True)

            # Encode with VAE
            vertices_static = vertices[:, :1].repeat(1, vertices.shape[1], 1, 1)
            num_traj = max(512, vertices.shape[2] // 8) if opt.num_traj < 0 else opt.num_traj
            x_start = vae_model(vertices_static, vertices[:, 0], faces=faces, valid_mask=valid_mask, adj_matrix=adj_matrix_nhops, num_traj=num_traj, just_encode=True)
            if opt.rescale:
                x0_start = (x_start[:, :, :opt.x0_channels] - x0_mean) / x0_std
                x1_start = (x_start[:, :, opt.x0_channels:opt.x0_channels+opt.x1_channels] - x1_mean) / x1_std
                xt_start = (x_start[:, :, -opt.xt_channels:] - xt_mean) / xt_std
                x_start = torch.cat([x0_start, x1_start, xt_start], dim=-1)
            
            # RF Model kwargs
            model_kwargs = dict(vid_embed=vid_embed)
            x0 = x_start[..., :opt.x0_channels]
            if opt.no_jump:
                x1 = x_start[..., opt.x0_channels:opt.x0_channels+opt.x1_channels]
            else:
                x1 = None

            # RF sampling
            print("Start RF sampling...")
            samples = rf_sample_vc_misalign(
                model=rf_model, 
                shape=x_start.shape, 
                model_kwargs=model_kwargs, 
                guidance_scale=opt.guidance_scale, 
                device=device, 
                x0=x0,
                x1=x1
            )
            print("RF sampling finished!!!")
            
            # DyMeshVAE decoding
            if opt.rescale:
                x0_start_s = samples[..., :opt.x0_channels] * x0_std + x0_mean
                x1_start_s = samples[..., opt.x0_channels:opt.x0_channels+opt.x1_channels] * x1_std + x1_mean
                xt_start_s = samples[..., -opt.xt_channels:] * xt_std + xt_mean 
                samples = torch.cat([x0_start_s, x1_start_s, xt_start_s], dim=-1)
            outputs = vae_model(vertices_static, vertices[:, 0], samples=samples, faces=faces, valid_mask=valid_mask, adj_matrix=adj_matrix_nhops, num_traj=opt.num_traj, just_decode=True)

            # Render & Export
            trajs = [outputs[0][:, idx].cpu()*v_max+center for idx in all_indices]
            file_format = mesh_path.split('.')[-1]
            drive_mesh_and_render_with_pkl(
                mesh_objects, 
                trajs, 
                "{}/{}".format(video_save_dir, video_name),
                export_format=file_format if opt.export else None,
                just_export=opt.just_export
            )
            
            if opt.render_frames:
                drive_mesh_and_render_with_pkl_frames(
                    mesh_objects, trajs, 
                    # "{}/{}".format(video_save_dir, video_name.split("/")[-1].split(".")[0]), 
                    "{}/{}".format(video_save_dir, video_name+'_azi0_ele0'),
                    azi=0.0
                )

                drive_mesh_and_render_with_pkl_frames(
                    mesh_objects, trajs, 
                    # "{}/{}".format(video_save_dir, video_name.split("/")[-1].split(".")[0]), 
                    "{}/{}".format(video_save_dir, video_name+'_azi90_ele15'),
                    azi=90.0,
                    ele=15.0
                )

                drive_mesh_and_render_with_pkl_frames(
                    mesh_objects, trajs, 
                    # "{}/{}".format(video_save_dir, video_name.split("/")[-1].split(".")[0]), 
                    "{}/{}".format(video_save_dir, video_name+'_azi180_ele15'),
                    azi=180.0,
                    ele=15.0
                )

                drive_mesh_and_render_with_pkl_frames(
                    mesh_objects, trajs, 
                    # "{}/{}".format(video_save_dir, video_name.split("/")[-1].split(".")[0]), 
                    "{}/{}".format(video_save_dir, video_name+'_azi270_ele15'),
                    azi=270.0,
                    ele=15.0
                )

if __name__ == '__main__':
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./test_data/meshes")
    parser.add_argument("--video_data_dir", type=str, default="./test_data/videos")
    parser.add_argument("--mesh_list", nargs="+", default=None)
    parser.add_argument("--video_list", nargs="+", default=None)
    parser.add_argument("--vae_dir", type=str, default="./ckpts/dvae")
    parser.add_argument("--rf_model_dir", type=str, default="./ckpts/rf_model")
    parser.add_argument("--wan_model_dir", type=str, default="./ckpts/Wan2.2-TI2V-5B")
    parser.add_argument("--json_dir", type=str, default="./ckpts/dvae_factors")
    parser.add_argument("--rf_exp", type=str, default="rdmeshdit")
    parser.add_argument("--rf_epoch", type=str, default='f')
    parser.add_argument("--video_save_dir", type=str, default="./output_videos")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--num_traj", type=int, default=-1)
    parser.add_argument("--rescale", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default="vc", choices=["tc", "vc", "vc_flext"])
    parser.add_argument("--video_width", type=int, default=256)
    parser.add_argument("--num_test", type=int, default=10)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--num_hops", type=int, default=4)
    parser.add_argument("--alpha_hops", type=float, default=0.5)
    parser.add_argument("--hop_mode", default="band", choices=["band", "single"])
    parser.add_argument("--dit_layers", type=int, nargs="+", default=[10])
    parser.add_argument("--testset", default="rdmesh")
    parser.add_argument("--samp_ratio", type=int, default=1)
    parser.add_argument("--num_t", type=int, default=-1)
    parser.add_argument("--ori_samp", action="store_true")
    parser.add_argument("--just_export", action="store_true")
    parser.add_argument("--no_jump", action="store_true")
    parser.add_argument("--render_frames", action="store_true")
    
    opt = parser.parse_args()

    opt.rescale = True

    main(opt)

'''


'''


   
