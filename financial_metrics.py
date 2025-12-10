import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def interative_plot(df, title):
    fig = px.line(title=title)
    for i in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y=df[i], name=i)

    return fig


def plot_reggeration_line(stocks_daily_return):
    plt.figure(figsize=(8, 8))
    for i in stocks_daily_return.columns:
        if i != 'Date' and i != 'sp500':
            stocks_daily_return.plot(kind='scatter', x='sp500', y=i, figsize=(8, 8))
            b, a = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[i], 1)
            # Plot the regression line
            plt.plot(stocks_daily_return['sp500'], b * stocks_daily_return['sp500'] + a, '-', color='r')
            plt.title(f'{i} vs sp500')
            plt.xlabel('sp500')
            plt.ylabel(i)
    plt.show()
    return None
def calculate_beta_alpha(stocks_daily_return):
    beta = {}
    alpha = {}
    for i in stocks_daily_return.columns:
        if i != 'Date' and i != 'sp500':
            b, a = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[i], 1)
            beta[i] = b
            alpha[i] = a
    return beta, alpha