<div align="center">
  <img src="https://github.com/R-DMesh/R-DMesh.github.io/blob/main/assets/logo.png" width="150px">

</div>

<div align="center">
  <h1>R-DMesh: Video-Guided 3D Animation via Rectified Dynamic Mesh Flow (SIGGRAPH 2026)</h1>
  
  
</div>




<div align="center">

Zijie Wu<sup>1,2</sup>, Lixin Xu<sup>2</sup>, Puhua Jiang<sup>2</sup>, Sicong Liu<sup>2</sup>, Chunchao Guo<sup>2</sup>, Xiang Bai<sup>1</sup> <br>
<sup>1</sup>Huazhong University of Science and Technology (HUST), <sup>2</sup>Tencent Hunyuan

<a href="https://r-dmesh.github.io/"><img src='https://img.shields.io/badge/Project-RDMesh-brightgreen?logo=github' alt='Project'></a>
<a href="https://arxiv.org/abs/"><img src='https://img.shields.io/badge/arXiv-RDMesh-B31B1B?logo=arxiv' alt='Paper PDF'></a>
<a href="https://youtu.be/xBrMjPnH3_w"><img src='https://img.shields.io/badge/Video-Demo-FF0000?logo=youtube' alt='Video'></a>
<a href="https://huggingface.co/JarrentWu/R-DMesh"><img src='https://img.shields.io/badge/HuggingFace-Model Weights-yellow?logo=huggingface' alt='Hugging Face Weights'></a>
<a href=""><img src='https://img.shields.io/badge/Google%20Drive-Model Weights-blue?logo=googledrive&logoColor=white' alt='Download from Google Drive'></a>

</div>

![Demo GIF](https://raw.githubusercontent.com/R-DMesh/R-DMesh.github.io/main/assets/teaser.gif)

## 📖 Overview

We present **R-DMesh**: a unified video-guided 4D mesh generation framework that tackles the long-overlooked pose misalignment dilemma. Given a static mesh and a reference video with arbitrary initial poses, our method automatically rectifies the mesh to the video's starting state and generates high-fidelity, temporally consistent animations. Beyond video-driven animation, R-DMesh naturally supports a wide range of downstream applications, including pose retargeting, motion retargeting, and holistic 4D generation.


## 🔥 Latest News

* May 13, 2026: 👋 The checkpoint of **R-DMesh** has been released! Please give it a try!
* May 12, 2026: 👋 The training & inference code of **R-DMesh** has been released! The checkpoint will be released in a few days.
* Mar 28, 2026: 👋 **R-DMesh** has been accepted by [SIGGRAPH2026](https://s2026.siggraph.org/)! We will release the code asap. Please stay tuned for updates！

## 🔧 Preparation

### 1. Environment Setup

```bash
# Create conda environment
conda create -n rdmesh python=3.11
conda activate rdmesh

# Install torch
pip install torch==2.8.0 torchvision==0.23.0

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pretrained Models

Download the pretrained checkpoints from [🤗 HuggingFace](https://huggingface.co/JarrentWu/R-DMesh) and place them under `./ckpts/`:

```bash
# Option 1: Use huggingface-cli
huggingface-cli download JarrentWu/R-DMesh --local-dir ./ckpts

# Option 2: Manually download and organize
```

You also need to download [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) for video conditioning:
```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./ckpts/Wan2.2-TI2V-5B
```

### 3. Prepare Test Data

Place your input meshes and reference videos under `./test_data/`.

### 📂 Expected Directory Structure

After the steps above, your project should look like this:

```
R-DMesh/
├── test_data/
│   ├── meshes/                  # Input meshes (.glb or .fbx)
│   │   └── your_mesh.glb
│   │   └── your_mesh2.fbx   
│   └── videos/                  # Reference videos (.mp4)
│       └── your_video.mp4
└── ckpts/
    ├── dvae/                    # VAE checkpoints
    ├── rf_model/                # Rectified Flow (DiT) checkpoints
    ├── dvae_factor/             # VAE normalization factors
    └── Wan2.2-TI2V-5B/          # Wan video model
```

## 📖 Inference

### 🎬 Animate a Mesh with Reference Video

```bash
python test_drive.py \
    --mesh_list your_mesh.glb \
    --video_list your_video.mp4 \
    --rf_exp rdmeshdit --rf_epoch f \
    --num_hops 5 --alpha_hops 0.7 \
    --num_traj 4096 --guidance_scale 1.5 \
    --export
```

> 💡 The command above assumes the [default directory structure](#-expected-directory-structure) from the Preparation section.  
> If your files are placed elsewhere, specify the paths explicitly:

```bash
    --data_dir /your/path/to/meshes \
    --video_data_dir /your/path/to/videos \
    --vae_dir /your/path/to/dvae \
    --rf_model_dir /your/path/to/rf_model \
    --json_dir /your/path/to/dvae_factor \
    --wan_model_dir /your/path/to/Wan2.2-TI2V-5B
```

An example is as follows, run:
```bash
python test_drive.py \
    --mesh_list warrok_w_kurniawan.fbx \
    --video_list dance7.mp4 \
    --rf_exp rdmeshdit --rf_epoch f \
    --num_hops 5 --alpha_hops 0.7 \
    --num_traj 4096 --guidance_scale 1.5 \
    --export
```
Then, you will get the dynamic mesh fbx file and a frontal rendered video, the generated 4D asset should look like:

![Demo GIF](https://raw.githubusercontent.com/R-DMesh/R-DMesh.github.io/main/assets/warrok_demo.gif)

> ⚠️ **Note on custom driving videos:**  
> If you want to use your own video to drive the mesh, please first remove the background and replace it with pure black using tools such as [SAM 3](https://github.com/facebookresearch/sam3) (or other video matting / segmentation tools) **before** running inference. Videos with cluttered or non-black backgrounds may lead to degraded motion extraction and poor animation quality.

## 🏋️‍♂️ Training

The complete training pipeline consists of the following **6 stages**, which must be executed **sequentially**:

```
① Data Preparation  →  ② Train R-DMesh VAE  →  ③ Extract Video Latents  →  ④ Extract DMesh Latents  →  ⑤ Compute DMesh Feature Statistics  →  ⑥ Train R-DMesh DiT
```

| Stage | Step | Script | Output |
| :---: | :--- | :--- | :--- |
| ① | Data Preparation | `data_construction/` | Mesh / Video dataset |
| ② | Train R-DMesh VAE | `train_dvae.py` | VAE checkpoints |
| ③ | Extract Video Latents | `Wan2_2/save_vid_latents.py` | Video latents |
| ④ | Extract DMesh Latents | `save_dmesh_latents.py` | DMesh latents |
| ⑤ | Compute DMesh Feature Statistics | `test_vae_factor_misalign.py` | Mean / std JSON factors |
| ⑥ | Train R-DMesh DiT | `train_dit.py` | DiT checkpoints |

---

### ① Data Preparation

Please refer to the scripts and README in the `data_construction` folder to build your training / validation data. This part of the code will be released soon.

---

### ② Train R-DMesh VAE

Train the R-DMesh VAE that compresses dynamic meshes into a latent space. To be noted, we adopt **PLTA attention** from [AnimateAnyMesh++](https://arxiv.org/abs/2604.26917) for better performance.

```bash
torchrun --nproc_per_node=8 train_dvae.py \
    --data_dir /path/to/training/data \
    --val_data_dir /path/to/validation/data \
    --ckpts_dir /path/to/checkpoints \
    --log_dir ./logs/test \
    --exp test \
    --train_epoch 1000 --batch_size 32 --lr 1e-4 \
    --enc_depth 8 --dim 256 --max_length 4096 \
    --latent_dim 64 --latent_dim_x1 16 --num_t 64 \
    --num_hops 4 --hop_mode band --n_layers 2 \
    --sep_rec_loss --per_instance_loss \
    --validate --is_training
```


(Optional) After training, you can evaluate the reconstruction quality of the VAE using the [Test R-DMesh VAE](#test-r-dmesh-vae) script in the Evaluation section.

---

### ③ Extract Video Latents

Extract latent features from reference videos using the pretrained video model ([Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)). These latents will serve as the conditioning signal for DiT training.

```bash
torchrun --nproc_per_node=8 save_vid_latents.py \
    --data_dir /path/to/mesh/data \
    --video_data_dir /path/to/video/data \
    --checkpoint_dir /path/to/pretrained/models \
    --output_dir /path/to/output/latents \
    --batch_size 1 \
    --video_width 256 \
    --samp_layer 10
```

---

### ④ Extract DMesh Latents

Encode meshes into latent variables using the VAE trained in Stage ②. These latents will be used as the prediction target for DiT training.

```bash
torchrun --nproc_per_node=8 save_dmesh_latents.py \
    --dataset_dir /path/to/dataset \
    --ckpt_dir /path/to/checkpoints \
    --output_dir /path/to/output/latents \
    --exp your_experiment_name \
    --epoch which_epoch \
    --max_length 8192 --batch_size 16 \
    --num_t 64 --num_hops 4 --num_traj 512
```

---

### ⑤ Compute DMesh Feature Statistics

Compute the per-channel **mean** and **standard deviation** of the DMesh latents produced by the VAE. These statistics are used to normalize (rescale) the latents so that their distribution is well-conditioned for DiT training. The resulting factors are saved as JSON files and later consumed by `train_dit.py` via the `--json_dir` argument.

```bash
torchrun --nproc_per_node=8 test_vae_factor_misalign.py \
    --data_dir /path/to/training/data \
    --vae_dir /path/to/dvae/checkpoints \
    --vae_exp your_vae_experiment \
    --vae_epoch which_epoch \
    --max_length max_vertex_count --batch_size 16 \
    --num_hops 4 --num_t 64
```

---

### ⑥ Train R-DMesh DiT

Train the conditional Diffusion Transformer using the video latents from Stage ③, the DMesh latents from Stage ④, and the normalization factors from Stage ⑤.

```bash
torchrun --nproc_per_node=8 train_dit.py \
    --data_dir /path/to/latent/data \
    --latent_data_dir /path/to/video/latents \
    --save_dir /path/to/checkpoints \
    --log_dir ./logs/test \
    --json_dir ./dvae_factors \
    --dvae_dir /path/to/dvae/checkpoints \
    --vae_exp your_vae_experiment \
    --vae_epoch which_epoch \
    --exp test \
    --batch_size 16 --max_length max_vertex_count --train_epoch 500 --lr 1e-4 \
    --mode vc --dit_layers 10 --cond_drop_prob 0.1 \
    --rescale
```

---

## 🔍 Evaluation

### Test R-DMesh VAE

Evaluate the reconstruction quality of a trained R-DMesh VAE.

```bash
python test_dvae.py \
    --dataset_dir /path/to/test/data \
    --ckpt_dir /path/to/checkpoints \
    --exp your_experiment_name \
    --epoch which_epoch \
    --max_length 4096 \
    --num_hops 4 --alpha_hops 0.5 \
    --render
```
    
## 📚 Citation

If you find our work interesting or helpful for your research, please consider citing:
```bibtex
@article{wu2026rdmesh,
  title={R-DMesh: Video-Guided 3D Animation via Rectified Dynamic Mesh Flow},
  author={Wu, Zijie and Xu, Lixin and Jiang, Puhua, and Liu, Sicong and Guo, Chunchao and Bai, Xiang},
  journal={arXiv preprint arXiv:},
  year={2026}
}
```

Please also consider citing [AnimateAnyMesh](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_AnimateAnyMesh_A_Feed-Forward_4D_Foundation_Model_for_Text-Driven_Universal_Mesh_ICCV_2025_paper.pdf) and [AnimateAnyMesh++](https://arxiv.org/abs/2604.26917), which inspired this work and provided techniques adopted in R-DMesh.
```bibtex
@inproceedings{wu2025animateanymesh,
  title={Animateanymesh: A feed-forward 4d foundation model for text-driven universal mesh animation},
  author={Wu, Zijie and Yu, Chaohui and Wang, Fan and Bai, Xiang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13557--13568},
  year={2025}
}
@article{wu2026animateanymesh++,
  title={AnimateAnyMesh++: A Flexible 4D Foundation Model for High-Fidelity Text-Driven Mesh Animation},
  author={Wu, Zijie and Yu, Chaohui and Wang, Fan and Bai, Xiang},
  journal={arXiv preprint arXiv:2604.26917},
  year={2026}
}
```

## 🙏 Acknowledgments

Our code references some great repos, which are [AnimateAnyMesh](https://github.com/JarrentWu1031/AnimateAnyMesh), [AnimateAnyMesh++](https://github.com/JarrentWu1031/AnimateAnyMesh-pp) and [Wan2_2](https://github.com/Wan-Video/Wan2.2). We thank the authors for their excellent works! <br>
