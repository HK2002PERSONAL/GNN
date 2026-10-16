import torch

def z_score(y: torch.Tensor) -> torch.Tensor:
    """标准 z-score"""
    return (y - y.mean()) / (y.std() + 1e-8)