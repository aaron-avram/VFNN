from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np

def get_data(start: str = '2001-01-01', end: str = datetime.today().date().strftime('%Y-%m-%d')):
    daily = yf.download(tickers='^SPX', start=start, end=end, interval='1d')
    close = np.array(daily[['Close']]).flatten()
    opn = np.array(daily[['Open']]).flatten()
    high = np.array(daily[['High']]).flatten()
    low = np.array(daily[['Low']]).flatten()

    vol, retr = _calc_stats(close, high, low, opn)

    df = pd.DataFrame({
        'date': daily.index,
        'volatility': vol,
        'returns': retr
    })
    return df

def _calc_stats(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, opn: pd.DataFrame) -> pd.DataFrame:
    u = np.log(np.divide(high, opn))
    d = np.log(np.divide(low, opn))
    c = np.log(np.divide(close, opn))

    vol = 0.511 * ((u - d)**2) - 0.019 * (c * (u + d) - 2 * u * d) - 0.383 * (c ** 2)
    retr = np.log(close[1:] / close[:-1])
    retr = np.insert(retr, 0, np.nan) # For shape alignment

    return vol, retr

if __name__ == '__main__':
    get_data()