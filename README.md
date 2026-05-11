<div align="center">
  <img src="https://github.com/R-DMesh/R-DMesh.github.io/blob/main/assets/logo.png" width="150px">

</div>

<div align="center">
  <h1>R-DMesh: Video-Guided 3D Animation via Rectified Dynamic Mesh Flow (Siggraph 2026)</h1>
  
  
</div>

<div align="center">

Zijie Wu<sup>1,2</sup>, Lixin Xu<sup>2</sup>, Puhua Jiang<sup>2</sup>, Sicong Liu<sup>2</sup>, Chunchao Guo<sup>2</sup>, Xiang Bai<sup>1</sup> <br>
<sup>1</sup>Huazhong University of Science and Technology (HUST), <sup>2</sup>Tencent Hunyuan

<a href="https://r-dmesh.github.io/"><img src='https://img.shields.io/badge/Project-RDMesh-brightgreen?logo=github' alt='Project'></a>
<a href="https://arxiv.org/abs/"><img src='https://img.shields.io/badge/arXiv-RDMesh-B31B1B?logo=arxiv' alt='Paper PDF'></a>
<a href="https://youtu.be/xBrMjPnH3_w"><img src='https://img.shields.io/badge/Video-Demo-FF0000?logo=youtube' alt='Video'></a>
<a href=""><img src='https://img.shields.io/badge/HuggingFace-Model Weights-yellow?logo=huggingface' alt='Hugging Face Weights'></a>
<a href=""><img src='https://img.shields.io/badge/Google%20Drive-Model Weights-blue?logo=googledrive&logoColor=white' alt='Download from Google Drive'></a>

</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/R-DMesh/R-DMesh.github.io/main/assets/teaser.png" alt="R-DMesh Teaser" width="100%">
</div>

## 📖 Overview

Video-guided 3D animation holds immense potential for content creation, offering intuitive and precise control over dynamic assets. However, practical deployment faces a critical yet frequently overlooked hurdle: the pose misalignment dilemma. In real-world scenarios, the initial pose of a user-provided static mesh rarely aligns with the starting frame of a reference video. Naively forcing a mesh to follow a mismatched trajectory inevitably leads to severe geometric distortion or animation failure. To address this, we present Rectified Dynamic Mesh (R-DMesh), a unified framework designed to generate high-fidelity 4D meshes that are ``rectified'' to align with video context. Unlike standard motion transfer approaches, our method introduces a novel VAE that explicitly disentangles the input into a conditional base mesh, relative motion trajectories, and a crucial rectification jump offset. This offset is learned to automatically transform the arbitrary pose of the input mesh to match the video's initial state before animation begins. We process these components via a Triflow Attention mechanism, which leverages vertex-wise geometric features to modulate the three orthogonal flows, ensuring physical consistency and local rigidity during the rectification and animation process. For generation, we employ a Rectified Flow-based Diffusion Transformer conditioned on pre-trained video latents, effectively transferring rich spatio-temporal priors to the 3D domain. To support this task, we construct Video-RDMesh, a large-scale dataset of over 500k dynamic mesh sequences specifically curated to simulate pose misalignment. Extensive experiments demonstrate that R-DMesh not only solves the alignment problem but also enables robust downstream applications, including pose retargeting and holistic 4D generation. 


## 🔥 Latest News

* May 12, 2026: 👋 The training & inference code of **R-DMesh** has been released! The checkpoint will be released in a few days.
* Mar 28, 2026: 👋 **R-DMesh** has been accepted by [Siggraph2026](https://s2026.siggraph.org/)! We will release the code asap. Please stay tuned for updates！

## 🔧 Preparation

```bash
# Create conda environment
conda create -n rdmesh python=3.11
conda activate rdmesh

# Install dependencies
pip install -r requirements.txt
```

## 📖 Inference

### 🎬 Animate a Mesh with Reference Video (Main Demo)

Animate your 3D mesh using a reference video:

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
