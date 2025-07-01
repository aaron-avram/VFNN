"""
Init file for the net class
"""
from .modules import VFNN, torch, nn
from .train import sgd, adam
from .loss import MAE, MAPE, RMSE

__all__ = ["VFNN", "sgd", "adam", "torch", "nn", "MAE", "MAPE", "RMSE"]
