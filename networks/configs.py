import torch
from .rdmeshdit import RDMeshDiT

def model_from_config(config, device):
    config = config.copy()
    return RDMeshDiT(device=device, dtype=torch.float32, **config)