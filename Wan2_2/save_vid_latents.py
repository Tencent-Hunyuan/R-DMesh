import argparse
import logging
import os
import pickle
from typing import List
import csv
import random
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

# WAN
import wan
from wan.configs import WAN_CONFIGS

LANCZOS_FILTER = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS

# -----------------------------
# DDP Utilities
# -----------------------------
def setup_ddp():
    """Initialize distributed data parallel setup"""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def is_main_process() -> bool:
    """Check if current process is the main process"""
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def barrier():
    """Synchronize all processes"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def setup_logger(is_main: bool):
    """Setup logger with appropriate level"""
    level = logging.INFO if is_main else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
        tensor_dict: Dictionary containing latent, idx, etc. to save
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build output file path
    output_path = os.path.join(output_dir, original_bin_name)
    
    # Prepare data for saving
    save_dict = {}
    for k, v in tensor_dict.items():
        if torch.is_tensor(v):
            save_dict[k] = v.detach().cpu()
        else:
            save_dict[k] = v
    
    # Atomic write
    tmp_path = output_path + ".tmp"
    atomic_write_pickle(save_dict, tmp_path, output_path)
    
    return output_path

# -----------------------------
# Image loading utilities
# -----------------------------
def load_video_frames(folder_path: str, target_frames: int=65, video_width: int=256):
    """
    Load PNG sequence frames with center crop and resize
    
    Returns: video_tensor
    - video_tensor: C,T,H,W where T=target_frames, H=W=video_width, pixel range [-1, 1]
    """
    names = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])

    assert len(names) >= target_frames, f"Video folder {folder_path} has less than {target_frames} frames"
    
    # Directly sample first target_frames frames
    final_names_to_load = names[:target_frames]
    
    if len(final_names_to_load) != target_frames:
        raise RuntimeError(f"Logic error: final file list length is {len(final_names_to_load)}, expected {target_frames} in {folder_path}")
    
    # Load images and create tensor
    tensors = []
    for fn in final_names_to_load:
        p = os.path.join(folder_path, fn)
        with Image.open(p) as img:
            if img.mode == 'RGBA':
                bg = Image.new('RGB', img.size, (0, 0, 0))
                bg.paste(img, (0, 0), img)
                img = bg
            else:
                img = img.convert('RGB')
            
            # Center crop to largest square area before resize
            width, height = img.size
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            img = img.crop((left, top, right, bottom))
            
            if img.size != (video_width, video_width):
                img = img.resize((video_width, video_width), resample=LANCZOS_FILTER)
            
            t = TF.to_tensor(img).sub_(0.5).div_(0.5)
            tensors.append(t.unsqueeze(1))
    
    video_tensor = torch.cat(tensors, dim=1)  # C,target_frames,256,256
    return video_tensor

def read_filenames_to_set(filepath: str) -> set:
    """
    Read filenames from file and process into a set
    - For .txt files: each line is treated as a filename
    - For .csv files: skip header, take first column as source
    
    Args:
        filepath: Path to .txt or .csv file
    
    Returns:
        set: Processed filename set (e.g., {'name1.bin', 'name2.bin'})
    
    Raises:
        ValueError: If file type is not .txt or .csv
        FileNotFoundError: If file path doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: File not found '{filepath}'")

    filenames = set()
    
    # Use 'utf-8-sig' to handle potential BOM
    with open(filepath, 'r', encoding='utf-8-sig', newline='') as f:
        if filepath.lower().endswith('.csv'):
            # CSV file processing
            reader = csv.reader(f)
            try:
                next(reader)  # Skip header
            except StopIteration:
                return set()  # Empty file
                
            for row in reader:
                if not row:  # Skip empty rows
                    continue
                base_name = row[0].split('.mp4')[0]
                filenames.add(base_name + ".bin")

        elif filepath.lower().endswith('.txt'):
            # TXT file processing
            for line in f:
                stripped_line = line.strip()
                if stripped_line:  # Skip empty lines
                    filenames.add(stripped_line)
        else:
            raise ValueError(f"Unsupported file type: '{filepath}'. Only '.txt' and '.csv' files are supported.")
            
    return filenames

# -----------------------------
# Dataset
# -----------------------------
class DyMeshDataset(Dataset):
    def __init__(self, data_dir: str, video_data_dir: str, num_t: int = 64, video_width: int = 256, 
                 limit: int = -1, file_txt: str = None, file_txt_processed: str = None):
        self.data_dir = data_dir
        self.video_data_dir = video_data_dir
        self.num_t = num_t
        self.video_width = video_width

        # Step 1: Get candidate file set
        candidates = set()
        if file_txt is not None:
            logging.info(f"Reading candidate filenames from '{file_txt}'...")
            candidates = read_filenames_to_set(file_txt)
        else:
            logging.info(f"Scanning candidate filenames from directory '{data_dir}'...")
            if os.path.exists(data_dir):
                candidates = set(os.listdir(data_dir))
            else:
                logging.error(f"Data directory '{data_dir}' does not exist.")
                raise FileNotFoundError(f"Data directory '{data_dir}' does not exist")

        # Step 2: Get exclusion set
        exclusions = set()
        if file_txt_processed is not None:
            logging.info(f"Reading processed filenames from '{file_txt_processed}'...")
            exclusions = read_filenames_to_set(file_txt_processed)

        # Step 3: Calculate difference and sort
        if exclusions:
            final_set = candidates - exclusions
            actual_excluded = len(candidates & exclusions)
            logging.info(f"Initial candidates: {len(candidates)}, to exclude: {len(exclusions)}")
            logging.info(f"Actually excluded: {actual_excluded}, remaining: {len(final_set)}")
        else:
            final_set = candidates
            logging.info(f"No exclusions, found {len(final_set)} items to process")

        files = sorted(list(final_set))

        # Apply limit
        if limit is not None and limit > 0:
            files = files[:limit]

        # Match files with video data
        logging.info(f"Scanning video data directory '{video_data_dir}'...")
        
        try:
            video_folder_set = set(os.listdir(video_data_dir))
        except FileNotFoundError:
            logging.warning(f"Video directory {video_data_dir} does not exist!")
            video_folder_set = set()

        matched = []
        for f in files:
            stem = f[:-4] 
            if stem in video_folder_set:
                matched.append(f)

        if not matched:
            raise RuntimeError(f"No valid pairs found in {data_dir} and {video_data_dir}")
        
        logging.info(f"Matching complete: {len(matched)}/{len(files)} files matched with video data")

        self.files = matched
        
        F = num_t
        ow = oh = video_width
        self.seq_len = (F // 4 + 1) * (oh // 16) * (ow // 16) // (2 * 2)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        bin_name = self.files[idx]
        
        try:
            stem = bin_name[:-4]
            bin_path = os.path.join(self.data_dir, bin_name)
            video_frames_path = os.path.join(self.video_data_dir, stem)
            
            videos = load_video_frames(video_frames_path, target_frames=self.num_t+1, video_width=self.video_width)
            videos = torch.cat([videos[:, 1:2], videos[:, 1:]], dim=1)

            # Validate data
            if videos is None:
                raise ValueError("Loaded video is None")

            assert videos.shape[1] % 4 == 1

            return {
                "bin_name": bin_name,
                "bin_path": bin_path,
                "videos": videos,
                "seq_len": self.seq_len,
            }

        except Exception as e:
            logging.warning(f"Error loading index {idx}, file: {bin_name}. Skipping sample.")
            logging.warning(f"Reason: {e}")
            return None

# -----------------------------
# Batched output utilities
# -----------------------------
def split_batched_dict(output_dict: dict, batch_size: int) -> List[dict]:
    """Split batched output dictionary into list of individual sample dictionaries"""
    results = [dict() for _ in range(batch_size)]
    
    for k, v in output_dict.items():
        if torch.is_tensor(v):
            assert v.shape[0] == batch_size, f"Key {k} tensor first dim != batch"
            for i in range(batch_size):
                results[i][k] = v[i].detach().cpu()
        elif isinstance(v, (list, tuple)):
            assert len(v) == batch_size, f"Key {k} list/tuple len != batch"
            for i in range(batch_size):
                results[i][k] = v[i]
        else:
            for i in range(batch_size):
                results[i][k] = v[i]
      
    return results

def collate_fn_skip_none(batch):
    """Collate function that skips None items"""
    # Filter out None items
    batch = [item for item in batch if item is not None]
    
    # Return None if batch is empty
    if len(batch) == 0:
        return None
    
    # Use default collate
    return default_collate(batch)

# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("DDP batched i2v processing (fixed 64x256x256)")
    p.add_argument("--data_dir", type=str, required=True, help="Source data directory")
    p.add_argument("--video_data_dir", type=str, required=True, help="Video frames directory")
    p.add_argument("--output_dir", type=str, required=True, help="Output latent directory")
    p.add_argument("--checkpoint_dir", type=str, required=True, help="Model checkpoint directory")
    p.add_argument("--model_key", type=str, default="ti2v-5B")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--no_ddp", action="store_true")
    p.add_argument("--file_txt", default=None, help="File list to process")
    p.add_argument("--file_txt_processed", default=None, help="Already processed file list")
    p.add_argument("--video_width", type=int, default=256)
    p.add_argument("--save_name", default=None, help="Save key name prefix")
    p.add_argument("--ori_samp", action="store_true", help="Use original sampling")
    p.add_argument("--samp_layer", type=int, default=10)
    return p.parse_args()

# -----------------------------
# Main function
# -----------------------------
def main():
    opt = parse_args()

    use_ddp = (not opt.no_ddp) and ("WORLD_SIZE" in os.environ) and (int(os.environ["WORLD_SIZE"]) > 1)
    if use_ddp:
        local_rank = setup_ddp()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    setup_logger(is_main_process())

    if is_main_process():
        logging.info(f"DDP={use_ddp}, world_size={world_size}, batch_size={opt.batch_size}")
        logging.info(f"Data directory: {opt.data_dir}")
        logging.info(f"Video directory: {opt.video_data_dir}")
        logging.info(f"Output directory: {opt.output_dir}")
        
        # Create output directory (only in main process)
        os.makedirs(opt.output_dir, exist_ok=True)
        logging.info(f"Output directory created/confirmed: {opt.output_dir}")

    dataset = DyMeshDataset(
        data_dir=opt.data_dir,
        video_data_dir=opt.video_data_dir,
        num_t=64,
        video_width=opt.video_width,
        limit=opt.limit,
        file_txt=opt.file_txt,
        file_txt_processed=opt.file_txt_processed,
    )
    
    sampler = DistributedSampler(dataset, shuffle=False) if use_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_skip_none,
    )

    barrier()

    cfg = WAN_CONFIGS[opt.model_key]
    wan_ti2v = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=opt.checkpoint_dir,
        device_id=local_rank,
        rank=rank,
        convert_model_dtype=True,
    )

    barrier()

    iterator = dataloader
    if is_main_process():
        iterator = tqdm(dataloader, desc="Processing", dynamic_ncols=True)

    torch.set_grad_enabled(False)
    with torch.no_grad():
        for batch in iterator:
            if batch is None:
                continue
            
            bin_names = batch["bin_name"]  # list[str] length B, only filenames
            videos = batch["videos"].to(device, non_blocking=True)  # B,C,64,256,256
            seq_len = int(batch["seq_len"][0].item() if torch.is_tensor(batch["seq_len"]) else batch["seq_len"][0])
            
            outputs = wan_ti2v.v2l_sample(
                videos=videos, seq_len=seq_len, samp_layers=[opt.samp_layer],
                save_name=opt.save_name, ori_samp=opt.ori_samp
            )

            # Process output and save to new files
            if isinstance(outputs, list):
                # Output is already split by sample
                assert len(outputs) == len(bin_names), "Output list length doesn't match batch size"
                
                for i, (bin_name, tdict) in enumerate(zip(bin_names, outputs)):
                    # Save to new file
                    output_path = save_latent_to_new_file(opt.output_dir, bin_name, tdict)
                    if is_main_process() and i == 0:
                        logging.info(f"Sample saved to: {output_path}")
                        
            elif isinstance(outputs, dict):
                # Output is batched dictionary, need to split
                bsz = videos.shape[0]
                split_dicts = split_batched_dict(outputs, bsz)
                
                for bin_name, tdict in zip(bin_names, split_dicts):
                    # Save to new file
                    output_path = save_latent_to_new_file(opt.output_dir, bin_name, tdict)
                    
            else:
                raise TypeError(f"Unsupported output type: {type(outputs)}")

    barrier()
    
    if use_ddp:
        dist.destroy_process_group()
        
    if is_main_process():
        logging.info("=" * 80)
        logging.info(f"All processing completed! Results saved to: {opt.output_dir}")
        logging.info("=" * 80)

if __name__ == "__main__":
    main()

'''

torchrun --nproc_per_node=8 \
save_vid_latents.py \
--data_dir /mnt/zw/zijiewu/Datasets/test_rdmesh_data/dmesh \
--video_data_dir /mnt/zw/zijiewu/Datasets/test_rdmesh_data/renderings_256/azi0 \
--checkpoint_dir /mnt/zw/zijiewu/pretrained_models/Wan2.2-TI2V-5B \
--batch_size 1 \
--video_width 256 \
--output_dir /mnt/zw/zijiewu/Datasets/test_rdmesh_data/layer_10/azi0 \
--samp_layer 10

'''