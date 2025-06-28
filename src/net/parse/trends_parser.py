"""
A file containing functions to query pytrends API and clean the retrieved data
"""

import time
from datetime import datetime, timedelta
from pytrends.request import TrendReq
import pandas as pd

def get_daily_trends_5yrs_single(keyword, overlap=30):
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
        except Exception as e:
            print(f"Failed to fetch {tf}: {e}")

        df = df.loc[~df.index.duplicated(keep='first')] # Throw out duplicates
        dfs.append(df)


        cur_start += delta
        time.sleep(1)
    normalize_windows(dfs, overlap)
    data = pd.concat(dfs)
    data = data[~data.index.duplicated(keep='first')]

    return data


def _normalize_overlapping(window1: pd.DataFrame, window2: pd.DataFrame, overlap=30) -> None:
    """
    Mutate the second dataframe so that it is normalized with reference to the first window
    """

    m1 = window1.iloc[-overlap:].mean()
    m2 = window2.iloc[:overlap].mean()
    ratio = (m1 / (m2 + 1e-12)).values[0]
    window2 *= ratio

def normalize_windows(windows: list[pd.DataFrame], overlap=30) -> None:
    """
    Mutate the dataframes in the list so that they are all normalized with respect
    to the first window
    """
    for w1, w2 in zip(windows, windows[1:]):
        _normalize_overlapping(w1, w2, overlap)
