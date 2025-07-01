"""
Init file for the net class
"""
from .modules import VFNN
from .train import sgd

__all__ = ["VFNN", "sgd"]
