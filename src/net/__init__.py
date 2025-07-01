"""
Init file for the net class
"""
from .modules import VFNN, torch, nn
from .train import sgd
from .loss import MAPE

__all__ = ["VFNN", "sgd", "torch", "nn", "MAPE"]
