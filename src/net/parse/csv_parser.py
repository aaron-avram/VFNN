from pathlib import Path
import pandas as pd

def parse_path(path: Path) -> list[pd.DataFrame]:
    dfs = []
    start_date = '2004-01-01'
    end_date = '2025-06-29'
    for f in path.glob('*.csv'):
        df = pd.read_csv(f)
        df['date'] = pd.to_datetime(df['date'])
        df[df.columns[1]] = df[df.columns[1]]
        df = interpolate_df(df, start_date, end_date)
        dfs.append(df)
    return dfs

def interpolate_df(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date']) # Ensure dates are in datetime format
    df = df.set_index('date') # Set date as index for interpolation

    full_index = pd.date_range(start_date, end_date, freq='D') # What to interpolate to
    df = df.reindex(full_index) # Add missing indices
    df.index.name = 'date'

    # Interpolate
    df = df.interpolate('linear')
    df = df.reset_index()

    return df

def merge_on_date(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    dfs = [df.set_index('date') for df in dfs]
    merged = pd.concat(dfs, axis=1, join='outer')
    return merged.reset_index()

if __name__ == '__main__':
    print(parse_path(Path('../../../data/trends'))[0])
