import streamlit as st
import pandas as pd
import yfinance as yf
def fetching_data(tickers_input,start_date,end_date):
    tickers = [ticker.strip() for ticker in tickers_input.split(",")]
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    sp500_data = yf.download("^GSPC", start=start_date, end=end_date)['Adj Close']
    data['sp500'] = sp500_data
    data.reset_index(inplace=True)
    return data
def Normalizing(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i] = x[i] / x[i][0]
    return x
def daily_return(df):
    df_daily_return=df.copy()
    for i in  df.columns[1:]:
        for j in range(1,len(df)):
           df_daily_return[i][j]=((df[i][j]-df[i][j-1])/df[i][j-1])*100
        df_daily_return[i][0] = 0
    return df_daily_return