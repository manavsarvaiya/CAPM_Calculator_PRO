
# CAPM Calculator Pro

# https://capm-calculator-pro.streamlit.app/

## Overview

The *CAPM Calculator Pro* is an interactive web application built with Streamlit that allows investors, students, and financial analysts to perform sophisticated CAPM analysis on any combination of stocks. The application calculates key financial metrics, visualizes risk-return relationships, and helps optimize portfolio allocation based on the Capital Asset Pricing Model framework

## Features

- Real-time Stock Data: Fetch current and historical data for any stock using Yahoo Finance API

- CAPM Calculations: Compute Beta, Alpha, Expected Returns, and other key metrics

- Interactive Visualizations: Create beautiful, interactive charts with Plotly

- Portfolio Analysis: Analyze custom portfolios with user-defined weights

- Risk Metrics: Calculate Sharpe Ratio, Volatility, Max Drawdown, and more

- Data Export: Download all analysis results as CSV files

- Responsive Design: Works seamlessly on desktop and mobile devices

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Technical Implementation](#technical-implementation)
- [CAPM Theory](#capm-theory)
- [Beta Interpretation](#beta-interpretation)
- [Sample Analysis](#sample-analysis)
- [Future Enhancements](#future-enhancements)

## Installation

### 1. Clone the repository

```
git clone https://github.com/yourusername/capm-calculator-pro.git
cd capm-calculator-pro
```

### 2. Create and activate a virtual environment

```
# For Mac/Linux
python3 -m venv capm_env
source capm_env/bin/activate

# For Windows
python -m venv capm_env
capm_env\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the application

```
streamlit run CAPM_Calculator.py
```


## Project Structure

```plaintext
CAPM-Calculator-Pro/
│
├── CAPM_Calculator.py               
├── helpers.py            
├── preprocessor.py       
├── requirements.txt
└── README.md   
```

## Usage Guide

### 1. Input Parameters

- Stock Tickers: Enter comma-separated ticker symbols (e.g., AAPL,MSFT,GOOGL)
- Date Range: Select start and end dates for analysis
- Market Index: Choose from S&P 500 (^GSPC), SPY, NASDAQ (QQQ), or others
- Risk-Free Rate: Adjust based on current Treasury bond yields
- Expected Market Return: Set your market return expectation (default: 7%)

### 2. Portfolio Weights

Assign custom weights to your portfolio:

- Use the sliders in the sidebar to allocate percentages
- Weights must sum to 100%
- Leave equal-weighted if no custom allocation is needed

### 3. Analysis Outputs

The application provides four main analysis tabs:

- Normalized Performance
   - Visual comparison of all assets normalized to start at 1.0
   - Performance relative to market benchmark
   - Total returns and CAGR calculations

- CAPM Results
   - Beta values for each stock
   - Alpha (excess return) calculations
   - Expected returns based on CAPM formula
   - R-squared values showing goodness of fit

- Portfolio Analysis
   - Portfolio Beta calculation
   - Expected portfolio return
   - Sharpe Ratio and volatility metrics
   - Visual comparison vs. market benchmark
   - Interactive pie chart for weight visualization

- Risk Metrics
   - Volatility (annualized standard deviation)
   - Sharpe and Sortino Ratios
   - Maximum Drawdown analysis
   - Risk-return scatter plots

### 4. Data Export

Export all analysis results as CSV files:
   - Raw price data
   - Daily returns
   - Normalized prices

## Technical Implementation

### Core Functions

```
# Data fetching with caching
@st.cache_data(ttl=3600)
def fetch_data(tickers, start, end, market_ticker='SPY'):
    # Downloads and processes stock data
    
# CAPM calculations
def calculate_statistics(returns_df, market_returns_col):
    # Computes Beta, Alpha, Sharpe Ratio, etc.
    
# Portfolio analysis
def analyze_portfolio(stats_dict, weights, risk_free_rate, expected_market_return):
    # Calculates portfolio metrics
```

### Key Libraries

- `Streamlit`: Web application framework

- `yfinance`: Financial data API

- `Plotly`: Interactive visualizations

- `Pandas`: Data manipulation and analysis

- `NumPy`: Numerical computations

- `Matplotlib`: Static plotting (fallback)


## CAPM Theory

### Formula:

The Capital Asset Pricing Model formula is: 

$$
E(R_i) = R_f + \beta_i \times [E(R_m) - R_f]
$$

**Where:**
- $E(R_i)$ = Expected return on investment i
- $R_f$ = Risk-free rate
- $\beta_i$ = Beta of investment i
- $E(R_m)$ = Expected market return
- $[E(R_m) - R_f]$ = Market risk premium

## Beta Interpretation

| Beta Value | Interpretation | Risk Profile |
|------------|----------------|--------------|
| **β > 1.5** | Highly sensitive to market | Aggressive |
| **1.0 < β ≤ 1.5** | More volatile than market | Growth |
| **β = 1.0** | Moves with the market | Neutral |
| **0.5 < β < 1.0** | Less volatile than market | Defensive |
| **β ≤ 0.5** | Minimally sensitive to market | Conservative |
| **β < 0** | Moves opposite to market | Hedge |


## Sample Analysis

Here's an example analysis for a tech portfolio (AAPL, MSFT, GOOGL) from 2023-01-01 to 2024-01-01:

| Metric | AAPL | MSFT | GOOGL | Portfolio |
|--------|------|------|-------|-----------|
| **Beta** | 1.28 | 1.15 | 1.22 | 1.22 |
| **Expected Return** | 8.4% | 7.8% | 8.1% | 8.1% |
| **Sharpe Ratio** | 1.25 | 1.18 | 1.21 | 1.21 |
| **Volatility** | 22.5% | 20.8% | 21.9% | 21.1% |


## Future Enhancements

- **Portfolio Optimization**: Extend the project to include Markowitz portfolio optimization.
- **Real-Time Data**: Integrate real-time stock data for live analysis.
- **Historical Risk-Free Rate**: Automate the fetching of historical risk-free rates.
