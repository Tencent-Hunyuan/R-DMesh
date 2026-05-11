<div align="center">
  <h1>R-DMesh: Video-Guided 3D Animation via Rectified Dynamic Mesh Flow</h1>
  
  <p>A powerful framework for dynamic mesh generation and animation</p>
</div>

## 📖 Overview

R-DMesh is a robust framework for dynamic mesh generation and animation, featuring advanced VAE architectures and diffusion models for high-quality mesh animation.

## 🚀 Features

- **Advanced VAE Architecture**: Multi-hop encoding with band mode for efficient mesh representation
- **Diffusion-based Animation**: Text-to-trajectory generation for dynamic mesh animation
- **Multi-modal Conditioning**: Support for video and text conditioning
- **High Performance**: Optimized for large-scale mesh processing

## 🔧 Installation

```bash
# Create conda environment
conda create -n rdmesh python=3.11
conda activate rdmesh

# Install dependencies
pip install -r requirements.txt
```

## 🏋️‍♂️ Training

### Train R-DMesh VAE
```bash
torchrun --nproc_per_node=8 train_dvae.py \
    --data_dir /path/to/training/data \
    --val_data_dir /path/to/validation/data \
    --ckpts_dir /path/to/checkpoints \
    --log_dir ./logs/test \
    --train_epoch 1000 --batch_size 32 --enc_depth 8 --dim 256 --max_length 4096 \
    --latent_dim 64 --latent_dim_x1 16 --num_t 64 --validate --is_training --lr 1e-4  \
    --num_hops 4 --hop_mode band --n_layers 2  \
    --sep_rec_loss \
    --per_instance_loss \
    --exp test
```

### Extract Video Latents
```bash
torchrun --nproc_per_node=8 save_vid_latents.py \
    --data_dir /path/to/mesh/data \
    --video_data_dir /path/to/video/data \
    --checkpoint_dir /path/to/pretrained/models \
    --batch_size 1 \
    --video_width 256 \
    --output_dir /path/to/output/latents \
    --samp_layer 10
```

### Extract DMesh Latents
```bash
torchrun --nproc_per_node=8 save_dmesh_latents.py \
    --dataset_dir /path/to/dataset \
    --ckpt_dir /path/to/checkpoints \
    --exp your_experiment_name \
    --epoch 330 \
    --max_length 8192 \
    --num_t 64 \
    --batch_size 16 \
    --num_hops 4 \
    --num_traj 512 \
    --output_dir /path/to/output/latents
```

### Train R-DMesh DiT
```bash
torchrun --nproc_per_node=8 train_dit.py \
    --data_dir /path/to/latent/data \
    --latent_data_dir /path/to/video/latents \
    --log_dir ./logs/test \
    --json_dir ./dvae_factors \
    --save_dir /path/to/checkpoints \
    --dvae_dir /path/to/dvae/checkpoints \
    --vae_exp your_vae_experiment \
    --vae_epoch 330 \
    --rescale \
    --batch_size 16 --max_length 8192 --train_epoch 500 --lr 1e-4 \
    --mode vc --dit_layers 10 --cond_drop_prob 0.1 \
    --exp test
```

## 📖 Usage

### Test R-DMesh VAE
```bash
python test_dvae.py \
    --dataset_dir /path/to/test/data \
    --ckpt_dir /path/to/checkpoints \
    --max_length 4096 \
    --exp your_experiment_name \
    --num_hops 4 --alpha_hops 0.5 --epoch 500 \
    --render
```

### Animate Mesh
```bash
python test_drive.py \
    --data_dir /path/to/meshes \
    --video_data_dir /path/to/reference/videos \
    --vae_dir /path/to/vae/checkpoints \
    --rf_model_dir /path/to/rf/checkpoints \
    --json_dir ./dvae_factors \
    --wan_model_dir /path/to/wan/model \
    --num_hops 5 --mode vc --alpha_hops 0.7 \
    --seed 666 --rf_epoch 223 --num_traj 4096 --dit_layers 10 \
    --rf_exp your_rf_experiment \
    --testset your_test_set --guidance_scale 1.5 --video_width 256 --num_test 10 \
    --mesh_list your_mesh.glb \
    --video_list your_video.mp4
```

## 📊 Configuration

Key parameters for customization:
- `--num_hops`: Number of hops for mesh encoding (default: 4)
- `--hop_mode`: Hop aggregation mode (band/full)
- `--max_length`: Maximum sequence length for processing
- `--num_traj`: Number of trajectories for animation
- `--guidance_scale`: Control strength for conditional generation

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Thanks to the open-source community for various libraries and tools
- Inspired by recent advances in mesh processing and diffusion models
