from pathlib import Path
import pandas as pd
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

def _normalize(tensor: torch.Tensor):
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True) + 1e-12
    new = (tensor - mean) / std
    return new, mean, std

def _rolling_window(tensor: torch.Tensor, context: int = 10) -> tuple[torch.Tensor]:
    data = tensor.unfold(dimension=0, size=context, step=1).permute(0, 2, 1)
    targets = tensor[context:, -2] # Get volatility (a bit clumsy)

    return data[:-1], targets

def build_data(trends: pd.DataFrame, stats: pd.DataFrame, delta: int = 3, context: int = 10) -> tuple[torch.Tensor]:
    downsampled = _downsample(_merge(trends, stats), delta)

    # Split
    b1 = int(downsampled.shape[0] * 0.8)
    splits = (downsampled[:b1], downsampled[b1:])

    # build splits
    train_temp, mean, std = _normalize(splits[0])
    x_train, y_train = _rolling_window(train_temp, context)
    x_test, y_test = _rolling_window((splits[1] - mean) / std, context)

    return x_train, y_train, x_test, y_test, mean, std

if __name__ == '__main__':
    path = Path('../../../data/trends')
    path_data = parse_path(path)
    _trends = merge_on_date(path_data)
    _stats = get_data()
    _data = _merge(_trends, _stats)
    print(_data[:100])
    tens = _downsample(_data)
