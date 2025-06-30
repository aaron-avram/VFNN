"""
A file containing functions to query pytrends API and clean the retrieved data
"""

from pathlib import Path
import time
from datetime import datetime, timedelta
from pytrends.request import TrendReq
import pandas as pd

def get_daily_trends_5yrs_single(keyword, overlap=30) -> pd.DataFrame:
    pytrends = TrendReq(hl='en-US', tz=360) # Create request object
    end = datetime.today()
    start = end - timedelta(days=5*365)

    delta = timedelta(days=90 - overlap)
    cur_start = start
    dfs = []

    while cur_start < end:
        # Create time interval
        cur_end = min(cur_start + timedelta(days=90), end)
        tf = f"{cur_start.strftime('%Y-%m-%d')} {cur_end.strftime('%Y-%m-%d')}"

        try:
            # Load data
            pytrends.build_payload([keyword], timeframe=tf)

            # Query data
            df = pytrends.interest_over_time()[[keyword]]

            df = df.loc[~df.index.duplicated(keep='first')] # Throw out duplicates
            dfs.append(df)
        except Exception as e:
            print(f"Failed to fetch {tf}: {e}")

        cur_start += delta
        time.sleep(10)
    dfs = normalize_windows(dfs, overlap)
    data = pd.concat(dfs)
    data = data[~data.index.duplicated(keep='first')]

    return data

def get_weekly_trends_single(keyword, overlap=1) -> pd.DataFrame:
    pytrends = TrendReq(hl='en-US', tz=360) # Req object
    end = datetime.today().date() - timedelta(days=(5-overlap)*365)
    start = datetime(2004, 1, 1) # When google trends started

    tf = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
    pytrends.build_payload([keyword], timeframe=tf)

    weekly_df = pytrends.interest_over_time()[[keyword]]
    daily_index = pd.date_range(start=weekly_df.index.min(), end=weekly_df.index.max(), freq='D')
    daily_df = weekly_df.reindex(daily_index) # Expand weekly data to daily
    daily_df = daily_df.interpolate(method='linear')

    return daily_df

def total_trends(keyword, daily_overlap=30, year_overlap=1):
    full_daily = get_daily_trends_5yrs_single(keyword, daily_overlap)
    inter_daily = get_weekly_trends_single(keyword, year_overlap)
    inter_daily = _normalize_overlapping(full_daily, inter_daily, year_overlap * 365)

    value1 = inter_daily.index.max() - timedelta(days=(year_overlap) * 365)
    value2 = value1 - timedelta(days=1)

    combined_df = pd.concat([
        inter_daily.loc[:value2].copy(),
        full_daily.loc[value1:].copy()
    ])

    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    return combined_df

def save_trend_to_csv(df: pd.DataFrame, keyword: str, output_dir: str="../../../data/trends") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Assuming df has a single unnamed column
    df.columns = [keyword]

    file_path = output_path /  f"{keyword}.csv"
    df.to_csv(file_path, index_label="date")


def _normalize_overlapping(window1: pd.DataFrame, window2: pd.DataFrame, overlap=30) -> pd.DataFrame:
    """
    Mutate the second dataframe so that it is normalized with reference to the first window
    """

    m1 = window1.iloc[-overlap:].mean()
    m2 = window2.iloc[:overlap].mean()
    ratio = (m1 / (m2 + 1e-12)).values[0]
    return window2 * ratio

def normalize_windows(windows: list[pd.DataFrame], overlap=30) -> pd.DataFrame:
    """
    Mutate the dataframes in the list so that they are all normalized with respect
    to the first window
    """
    new = [windows[0]]
    for w1, w2 in zip(windows, windows[1:]):
        new.append(_normalize_overlapping(w1, w2, overlap))
    return new

# For testing
if __name__ == '__main__':
    keywords = ['mobile', 'software', 'hardware', 'investing', 'manufacturing', 'business news']
    for _keyword in keywords:
        _df = total_trends(_keyword)
        save_trend_to_csv(_df, _keyword)
        print("Done: ", _keyword)
