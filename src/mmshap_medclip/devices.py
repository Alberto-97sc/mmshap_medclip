import torch

def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    return batch.to(device) if hasattr(batch, "to") else batch
