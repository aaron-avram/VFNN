import yfinance as yf
import pandas as pd
import numpy as np
from net.parse.csv_parser import interpolate_df

def get_data(start: str = '2004-10-19', end: str = '2015-07-24', scale=1e6):
    daily = yf.download(tickers='^SPX', start=start, end=end, interval='1d')
    close = np.array(daily[['Close']]).flatten()
    opn = np.array(daily[['Open']]).flatten()
    high = np.array(daily[['High']]).flatten()
    low = np.array(daily[['Low']]).flatten()

    vol, retr = _calc_stats(close, high, low, opn, scale)


    df = pd.DataFrame({
        'date': daily.index,
        'volatility': vol,
        'returns': retr
    })
    df = interpolate_df(df, start, end)
    df.fillna(0, inplace=True)
    return df

def _calc_stats(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, opn: pd.DataFrame, scale=1e6) -> pd.DataFrame:
    u = np.log(np.divide(high, opn))
    d = np.log(np.divide(low, opn))
    c = np.log(np.divide(close, opn))

    vol = (0.511 * ((u - d)**2) - 0.019 * (c * (u + d) - 2 * u * d) - 0.383 * (c ** 2)) * scale
    retr = np.log(close[1:] / close[:-1]) * scale
    retr = np.insert(retr, 0, 0) # For shape alignment

    return vol, retr

if __name__ == '__main__':
    print(get_data())