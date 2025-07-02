"""
Loss functions for models
"""
import torch

def MAE(preds: torch.Tensor, acc: torch.Tensor):
    """
    Mean absolute error
    """
    return torch.abs(preds - acc).mean()

def MAPE(preds: torch.Tensor, acc: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """
    Mean absolute percentage error
    """
    preds = preds.flatten() * std + mean
    acc = acc.flatten() * std + mean
    return 100 * torch.abs((preds - acc) / (torch.abs(acc))).mean()

def RMSE(preds, targets):
    """
    Root mean squared error
    """
    return torch.sqrt(torch.mean((targets - preds)**2))