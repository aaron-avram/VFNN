from functools import reduce
from pathlib import Path
import pandas as pd

def parse_path(path: Path) -> list[pd.DataFrame]:
    dfs = []
    start_date = '2004-01-01'
    end_date = '2025-06-29'
    for f in path.glob('*.csv'):
        df = pd.read_csv(f)
        df = interpolate_df(df, start_date, end_date)
        dfs.append(df)
    return dfs

def interpolate_df(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date']) # Ensure dates are in datetime format
    df.set_index('date') # Set date as index for interpolation

    full_index = pd.date_range(start_date, end_date, freq='D') # What to interpolate to
    df = df.reindex(full_index) # Add missing indices

    # Interpolate
    df = df.interpolate('linear')
    df = df.reset_index().rename(columns={'index': 'date'})

    return df

def merge_on_date(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    return reduce(lambda x, y: pd.merge(x, y, on='date', how='outer'), dfs)