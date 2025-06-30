from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np

def get_raw_data(start: str = '2001-01-01', end: str = datetime.today().date().strftime('%Y-%m-%d')):
    daily = yf.download(tickers='^SPX', start=start, end=end, interval='1d')
    close = daily[['Close']]
    opn = daily[['Open']]
    high = daily[['High']]
    low = daily[['Low']]

    vol, retr = calc_stats(close, high, low, opn)
    return vol, retr

def calc_stats(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, opn: pd.DataFrame) -> pd.DataFrame:
    u = np.log(high / opn)
    d = np.log(low / opn)
    c = np.log(close / opn)

    vol = 0.511 * (u - d)**2 - 0.019 * (c * (u + d) - 2 * u * d) - 0.383 * c ** 2
    retr = np.log(close / close.shift(1))

    return vol, retr

if __name__ == '__main__':
    get_raw_data()