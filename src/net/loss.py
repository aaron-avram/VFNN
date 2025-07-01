"""
Loss functions for models
"""
import torch

def MAE(preds: torch.Tensor, acc: torch.Tensor):
    """
    Mean absolute error
    """
    return torch.abs(preds - acc).mean()

def MAPE(preds: torch.Tensor, acc: torch.Tensor):
    """
    Mean absolute percentage error
    """
    return 100 * torch.abs((preds - acc) / acc).mean()