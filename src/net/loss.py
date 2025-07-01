"""
Loss functions for models
"""
import torch

def MAPE(preds: torch.Tensor, acc: torch.Tensor):
    """
    Mean absolute percentage error
    """
    percentage = torch.abs((preds - acc) / (acc + 1e-12))
    return percentage.mean()