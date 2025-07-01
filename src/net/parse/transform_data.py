from pathlib import Path
import pandas as pd
import numpy as np
import torch
from net.parse.csv_parser import merge_on_date, parse_path
from net.parse.finance_parser import get_data

def _merge(trends: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(trends, stats, how='outer', on='date')

def _downsample(df: pd.DataFrame, start: int = 1, delta: int=3) -> torch.Tensor:
    new_size = (len(df) - start) // delta
    features = df.shape[1] - 1
    out = torch.zeros(new_size, features)
    numerics = df.select_dtypes(include='number').values
    VIX_COL = df.columns.get_loc('volatility')
    RET_COL = df.columns.get_loc('returns')
    for i in range(new_size):
        chunk = torch.tensor(numerics[start + delta * i: start + delta * (i + 1)])
        for j in range(features):
            if j == VIX_COL:
                out[i, j] = torch.sqrt(torch.sum(chunk[:, j] ** 2))
            elif j == RET_COL:
                out[i, j] = torch.sum(chunk[:, j])
            else:
                out[i, j] = torch.mean(chunk[:, j])
    return out

def _normalize(tensor: torch.Tensor, k: int | float = float('inf')):
    if k == float('inf'):
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True) + 1e-12
        new = (tensor - mean) / std
    else:
        new = torch.zeros_like(tensor)
        for i in range(0, tensor.shape[0], k):
            end = min(i + k, tensor.shape[0])
            cur = tensor[i:end, :] # Alias for convenience
            mean = cur.mean(dim=0, keepdim=True)
            std = cur.std(dim=0, keepdim=True) + 1e-12

            new[i:end, :] = (cur - mean) / std
    return new

def _rolling_window(tensor: torch.Tensor, context: int = 10) -> tuple[torch.Tensor]:
    data = tensor.unfold(dimension=0, size=context, step=1).permute(0, 2, 1)
    targets = tensor[context:, -2] # Get volatility (a bit clumsy)

    return data[:-1], targets

def build_data(trends: pd.DataFrame, stats: pd.DataFrame, delta: int = 3, k: int = float('inf'), context: int = 10) -> tuple[torch.Tensor]:
    downsampled = _downsample(_merge(trends, stats), delta)

    # Split
    b1 = int(downsampled.shape[0] * 0.8)
    b2 = int(downsampled.shape[0] * 0.9)
    splits = (downsampled[:b1], downsampled[b1:b2], downsampled[b2:])
    normed = list(map(lambda x: _normalize(x, k), splits))

    # build splits
    x_train, y_train = _rolling_window(normed[0], context)
    x_test, y_test = _rolling_window(normed[1], context)
    x_dev, y_dev = _rolling_window(normed[2], context)

    return x_train, y_train, x_test, y_test, x_dev, y_dev

if __name__ == '__main__':
    path = Path('../../../data/trends')
    path_data = parse_path(path)
    _trends = merge_on_date(path_data)
    _stats = get_data()
    _data = _merge(_trends, _stats)
    tens = _downsample(_data)
