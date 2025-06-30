from pathlib import Path
import pandas as pd
import numpy as np
import torch
from net.parse.csv_parser import merge_on_date, parse_path
from net.parse.finance_parser import get_data

def to_tensor(trends: pd.DataFrame, stats: pd.DataFrame, delta: int=3) -> torch.Tensor:
    merged = pd.merge(trends, stats, how='outer', on='date')
    data = downsample(merged, delta)
    return data

def downsample(df: pd.DataFrame, start: int = 1, delta: int=3) -> torch.Tensor:
    new_size = (len(df) - start) // delta
    features = df.shape[1] - 1
    out = torch.zeros(new_size, features)
    numerics = df.select_dtypes(include='number').columns
    for j, col_name in enumerate(numerics):
        col = df[col_name]
        for i in range(start, len(df) // delta - 1, delta):
            if j == 26:
                out[i, j] = np.sqrt(sum(col[k] for k in range(i, i + delta)))
            elif j == 27:
                out[i, j] = sum(col[k] for k in range(i, i + delta))
            else:
                out[i, j] = sum(col[k] for k in range(i, i + delta)) / delta
    return out

if __name__ == '__main__':
    path = Path('../../../data/trends')
    path_data = parse_path(path)
    _trends = merge_on_date(path_data)
    _stats = get_data()
    _data = to_tensor(_trends, _stats)

