import pandas as pd
from sklearn.linear_model import LinearRegression
import os

def load_stock_data(symbol):
    file_path = f"data/{symbol}.csv"

    if not os.path.exists(file_path):
        raise ValueError("Stock data not available locally")

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Handle different column names
    close_col = [c for c in df.columns if 'Close' in c][0]
    df = df[[close_col]].rename(columns={close_col: 'Close'})

    return df

def train_and_predict(stock_symbol):
    df = load_stock_data(stock_symbol)

    # Feature engineering
    df['Prev_Close'] = df['Close'].shift(1)
    df['MA_3'] = df['Close'].rolling(3).mean()
    df['MA_7'] = df['Close'].rolling(7).mean()
    df['Return'] = df['Close'].pct_change()

    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df[['Prev_Close', 'MA_3', 'MA_7', 'Return']]
    y = df['Target']

    model = LinearRegression()
    model.fit(X, y)

    df['Predicted'] = model.predict(X)

    return df
