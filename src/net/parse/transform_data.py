from pathlib import Path
import pandas as pd
import numpy as np
import torch
from net.parse.csv_parser import merge_on_date, parse_path
from net.parse.finance_parser import get_data

def merge(trends: pd.DataFrame, stats: pd.DataFrame) -> torch.Tensor:
    return pd.merge(trends, stats, how='outer', on='date')

def downsample(df: pd.DataFrame, start: int = 1, delta: int=3) -> torch.Tensor:
    new_size = (len(df) - start) // delta
    features = df.shape[1] - 1
    out = torch.zeros(new_size, features)
    numerics = df.select_dtypes(include='number').columns
    for j, col_name in enumerate(numerics):
        col = df[col_name]
        for i in range(start, len(df) // delta - 1, delta):
            if j == 26:
                out[i, j] = np.sqrt(sum(col[k] ** 2 for k in range(i, i + delta)))
            elif j == 27:
                out[i, j] = sum(col[k] for k in range(i, i + delta))
            else:
                out[i, j] = sum(col[k] for k in range(i, i + delta)) / delta
    return out

def normalize(tensor: torch.Tensor, window: int | float = float('inf')):
    if window == float('inf'):
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True) + 1e-12
        new = (tensor - mean) / std
    else:
        new = torch.zeros_like(tensor)
        for i in range(0, tensor.shape[0], window):
            end = i + window
            cur = tensor[i:end, :] # Alias for convenience
            mean = cur.mean(dim=1, keepdim=True)
            std = cur.std(dim=1, keepdim=True) + 1e-12

            new[i:end, :] = (cur - mean) / std
    return new

def rolling_window(tens: torch.Tensor, context: int = 8):
    return
if __name__ == '__main__':
    path = Path('../../../data/trends')
    path_data = parse_path(path)
    _trends = merge_on_date(path_data)
    _stats = get_data()
    _data = merge(_trends, _stats)
    tens = downsample(_data)
