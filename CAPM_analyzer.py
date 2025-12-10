import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import traceback
import io

# Set page config
st.set_page_config(
    page_title="CAPM Calculator Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #93C5FD;
        margin: 0.5rem 0;
    }
    .stButton button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
    }
    .stButton button:hover {
        background-color: #2563EB;
    }
    .data-frame {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">📊 Capital Asset Pricing Model (CAPM) Calculator Pro</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #6B7280; margin-bottom: 2rem;'>
Calculate expected returns, assess risk, and optimize your portfolio using the CAPM framework
</div>
""", unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.markdown("### ⚙️ Input Parameters")
    
    # Stock selection
    ticker_input = st.text_input('**Stock Tickers (comma-separated)**', 'AAPL,MSFT,GOOGL')
    
    # Date range - using more recent dates for better data fetching
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('**Start Date**', datetime(2024, 1, 1))
    with col2:
        end_date = st.date_input('**End Date**', datetime(2024, 12, 10))
    
    # Advanced parameters
    with st.expander("⚙️ Advanced Parameters", expanded=True):
        risk_free_rate = st.slider('**Risk-Free Rate (%)**', 0.0, 10.0, 2.0, 0.1) / 100
        
        # Expected market return input
        expected_market_return = st.number_input(
            '**Expected Market Return (%)**',
            value=7.0,
            min_value=0.0,
            max_value=30.0,
            step=0.5,
            help="Long-term expected market return for CAPM calculations (S&P 500 historical average is ~7%)"
        ) / 100
        
        col3, col4 = st.columns(2)
        with col3:
            market_ticker = st.selectbox(
                '**Market Index**',
                ['SPY', 'QQQ', 'DIA', 'IWM'],  # REMOVED: '^GSPC'
                index=0
            )
        with col4:
            lookback_period = st.selectbox(
                '**Lookback Period**',
                ['1 Month', '3 Months', '6 Months', '1 Year', 'Custom'],
                index=3
            )
    
    # Portfolio weights
    st.markdown("---")
    st.markdown("### 🎯 Portfolio Weights")
    st.markdown("Assign custom weights to your stocks (must sum to 100%)")
    
    if ticker_input:
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
        if len(tickers) > 1:
            weights = {}
            total = 0
            for i, ticker in enumerate(tickers):
                weight = st.slider(f'{ticker} Weight (%)', 0, 100, 100//len(tickers), key=f"weight_{i}")
                weights[ticker] = weight / 100
                total += weight
            
            if total != 100:
                st.warning(f"⚠️ Weights sum to {total}%. Adjust to sum to 100%.")
                custom_weights = False
            else:
                st.success("✓ Weights sum to 100%")
                custom_weights = True
        else:
            custom_weights = False
            weights = {tickers[0]: 1.0} if tickers else {}
    else:
        custom_weights = False
        weights = {}

# Enhanced Function to fetch data
@st.cache_data(ttl=3600)
def fetch_data(tickers, start, end, market_ticker='SPY'):
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        if not ticker_list:
            st.error("⚠️ No tickers provided")
            return None
        
        # Add market ticker if not already in list
        if market_ticker not in ticker_list:
            ticker_list.append(market_ticker)
        
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        
        # Download data
        data = yf.download(
            ticker_list,
            start=start_str,
            end=end_str,
            progress=False,
            auto_adjust=True,
            group_by='ticker'
        )
        
        if data.empty:
            st.error("❌ No data returned from Yahoo Finance")
            return None
        
        # Process data - handle different column formats
        close_prices = pd.DataFrame()
        
        # Extract close prices from the downloaded data
        if isinstance(data.columns, pd.MultiIndex):
            # Data has MultiIndex columns (ticker, price type)
            for ticker in ticker_list:
                try:
                    # Try to get Close prices for each ticker
                    if (ticker, 'Close') in data.columns:
                        close_prices[ticker] = data[(ticker, 'Close')]
                    elif ticker in data.columns.get_level_values(0):
                        # Try to get any column for this ticker
                        ticker_data = data[ticker]
                        if 'Close' in ticker_data.columns:
                            close_prices[ticker] = ticker_data['Close']
                        else:
                            # If no Close column, use the first available column
                            if not ticker_data.empty:
                                close_prices[ticker] = ticker_data.iloc[:, 0]
                except Exception as e:
                    st.warning(f"⚠️ Could not get data for {ticker}: {str(e)}")
        else:
            # Data has simple columns - older yfinance format
            if 'Close' in data.columns:
                # Single ticker case
                close_prices[ticker_list[0]] = data['Close']
            else:
                # Multiple tickers without MultiIndex
                close_prices = data
        
        # Drop any columns that are all NaN
        close_prices = close_prices.dropna(axis=1, how='all')
        
        if close_prices.empty:
            st.error("❌ No valid price data after processing")
            return None
        
        # Ensure column names are strings
        close_prices.columns = [str(col) for col in close_prices.columns]
        
        return close_prices
        
    except Exception as e:
        st.error(f"❌ Error in fetch_data: {str(e)}")
        return None

# Enhanced function to calculate statistics
def calculate_statistics(returns_df, market_returns_col):
    stats_dict = {}
    
    for column in returns_df.columns:
        if column != market_returns_col:
            returns = returns_df[column].dropna()
            market_returns = returns_df[market_returns_col].dropna()
            
            if len(returns) > 1 and len(market_returns) > 1:
                # Beta calculation using regression
                beta, alpha = np.polyfit(market_returns, returns, 1)
                
                # R-squared
                correlation = np.corrcoef(returns, market_returns)[0, 1]
                r_squared = correlation ** 2
                
                # Sharpe Ratio (assuming risk-free rate of 0 for daily)
                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                
                # Volatility (annualized)
                volatility = returns.std() * np.sqrt(252)
                
                # Maximum Drawdown
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min()
                
                stats_dict[column] = {
                    'Beta': beta,
                    'Alpha': alpha,
                    'R-squared': r_squared,
                    'Sharpe Ratio': sharpe,
                    'Volatility (annual)': volatility,
                    'Max Drawdown': max_drawdown,
                    'Mean Return (daily)': returns.mean(),
                    'Std Dev (daily)': returns.std()
                }
    
    return stats_dict

# Function to convert dataframe to CSV for download
def convert_df_to_csv(df):
    return df.to_csv(index=True).encode('utf-8')

# Custom styling functions (to avoid matplotlib dependency for background_gradient)
def color_beta(val):
    if val > 1.5:
        return 'background-color: #ffcccc'  # Light red for high beta
    elif val > 1:
        return 'background-color: #ffe6cc'  # Light orange
    elif val > 0.5:
        return 'background-color: #ffffcc'  # Light yellow
    else:
        return 'background-color: #e6ffe6'  # Light green

def color_return(val, expected_market_return):
    try:
        if isinstance(val, str):
            # Remove % sign and convert to float
            val_float = float(val.strip('%')) / 100
        else:
            val_float = float(val)
        
        if val_float > expected_market_return:
            return 'background-color: #e6ffe6'  # Light green
        else:
            return 'background-color: #ffe6e6'  # Light red
    except:
        return ''

# Main execution
if st.button('🚀 Calculate CAPM Analysis', type='primary', use_container_width=True):
    if ticker_input and ticker_input.strip():
        with st.spinner('📊 Fetching data and performing analysis...'):
            try:
                # Fetch data
                df = fetch_data(ticker_input, start_date, end_date, market_ticker)
                
                if df is not None and not df.empty:
                    # Debug info
                    st.info(f"✅ Successfully fetched data for: {', '.join(df.columns.tolist())}")
                    
                    # Success message with metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📅 Trading Days", len(df))
                    with col2:
                        st.metric("📊 Number of Assets", len([col for col in df.columns if col != market_ticker]))
                    with col3:
                        st.metric("📈 Market Index", market_ticker)
                    
                    # DATA EXPORT SECTION
                    st.markdown("---")
                    with st.expander("📥 View & Export Raw Data", expanded=False):
                        col_export1, col_export2 = st.columns([3, 1])
                        
                        with col_export1:
                            st.markdown("#### 📋 Extracted Price Data")
                            st.markdown(f"*Showing {min(10, len(df))} of {len(df)} rows*")
                            
                            # Display dataframe with formatting
                            display_df = df.round(2)
                            st.dataframe(display_df.head(10), use_container_width=True)
                            
                            # Show data summary
                            st.markdown("#### 📊 Data Summary")
                            summary_df = pd.DataFrame({
                                'Ticker': df.columns,
                                'Start Price': df.iloc[0].round(2),
                                'End Price': df.iloc[-1].round(2),
                                'Total Return': ((df.iloc[-1] / df.iloc[0] - 1) * 100).round(2).astype(str) + '%',
                                'Missing Values': df.isnull().sum()
                            })
                            st.dataframe(summary_df, use_container_width=True)
                        
                        with col_export2:
                            st.markdown("#### 💾 Export Options")
                            
                            # Export price data
                            csv_price = convert_df_to_csv(df)
                            st.download_button(
                                label="📥 Download Price Data (CSV)",
                                data=csv_price,
                                file_name=f"stock_prices_{start_date}_{end_date}.csv",
                                mime="text/csv",
                                help="Download the raw price data as CSV"
                            )
                            
                            # Export returns data
                            returns_df = df.pct_change().dropna()
                            csv_returns = convert_df_to_csv(returns_df)
                            st.download_button(
                                label="📥 Download Returns Data (CSV)",
                                data=csv_returns,
                                file_name=f"stock_returns_{start_date}_{end_date}.csv",
                                mime="text/csv",
                                help="Download daily returns data as CSV"
                            )
                            
                            # Export normalized data
                            normalized_df = df.div(df.iloc[0])
                            csv_normalized = convert_df_to_csv(normalized_df)
                            st.download_button(
                                label="📥 Download Normalized Data (CSV)",
                                data=csv_normalized,
                                file_name=f"normalized_prices_{start_date}_{end_date}.csv",
                                mime="text/csv",
                                help="Download normalized price data (starting at 1.0) as CSV"
                            )
                            
                            st.markdown("---")
                            st.markdown("**Quick Stats:**")
                            st.write(f"Date Range: {df.index[0].date()} to {df.index[-1].date()}")
                            st.write(f"Data Points: {len(df)}")
                            st.write(f"Columns: {len(df.columns)}")
                    
                    # Tabs for different sections
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📈 Normalized Performance", 
                        "📊 CAPM Results", 
                        "🎯 Portfolio Analysis",
                        "📉 Risk Metrics"
                    ])
                    
                    with tab1:
                        st.markdown('<h3 class="sub-header">Normalized Price Performance</h3>', unsafe_allow_html=True)
                        
                        # Create normalized prices
                        normalized_df = df.div(df.iloc[0])
                        
                        # Check what tickers we actually have data for
                        available_tickers = [t for t in normalized_df.columns if t != market_ticker]
                        st.info(f"📈 Available tickers for chart: {', '.join(available_tickers) if available_tickers else 'None'}")
                        
                        # Create figure
                        fig_norm = go.Figure()
                        
                        # Add each stock as a line (excluding market ticker)
                        colors = px.colors.qualitative.Set3
                        for idx, ticker in enumerate(normalized_df.columns):
                            if ticker == market_ticker:
                                continue  # We'll add market separately
                            
                            fig_norm.add_trace(go.Scatter(
                                x=normalized_df.index,
                                y=normalized_df[ticker],
                                mode='lines',
                                name=ticker,
                                line=dict(color=colors[idx % len(colors)], width=2),
                                hovertemplate=f'{ticker}<br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                            ))
                        
                        # Add market index as dashed line if available - FIXED
                        if market_ticker in normalized_df.columns:
                            fig_norm.add_trace(go.Scatter(
                                x=normalized_df.index,
                                y=normalized_df[market_ticker],
                                mode='lines',
                                name=f'Market ({market_ticker})',
                                line=dict(dash='dash', color='black', width=2.5),
                                opacity=0.7,  # CORRECT: opacity at trace level
                                hovertemplate=f'Market ({market_ticker})<br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                            ))
                        
                        # Update layout
                        fig_norm.update_layout(
                            title="Normalized Price Performance (Base = 1.0)",
                            xaxis_title="Date",
                            yaxis_title="Normalized Price (Base = 1.0)",
                            height=500,
                            hovermode='x unified',
                            template='plotly_white',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,
                                title_text="Tickers"
                            )
                        )
                        
                        st.plotly_chart(fig_norm, use_container_width=True)
                        
                        # Performance comparison table
                        st.markdown("#### 📊 Performance Comparison")
                        
                        # Calculate performance metrics
                        performance_data = []
                        for ticker in normalized_df.columns:
                            total_return = (normalized_df[ticker].iloc[-1] - 1) * 100
                            
                            # Calculate CAGR if we have enough data
                            if len(normalized_df) > 1:
                                days = len(normalized_df)
                                cagr = ((normalized_df[ticker].iloc[-1] ** (252/days)) - 1) * 100
                            else:
                                cagr = 0
                            
                            performance_data.append({
                                'Ticker': ticker,
                                'Start Value': 1.00,
                                'End Value': round(normalized_df[ticker].iloc[-1], 3),
                                'Total Return': f"{total_return:.2f}%",
                                'CAGR (%)': round(cagr, 2)
                            })
                        
                        performance_df = pd.DataFrame(performance_data)
                        
                        # Color code the CAGR column
                        def color_cagr(val):
                            if isinstance(val, (int, float)):
                                color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                                return f'color: {color}; font-weight: bold'
                            return ''
                        
                        styled_perf = performance_df.style.applymap(color_cagr, subset=['CAGR (%)'])
                        st.dataframe(styled_perf, use_container_width=True)
                    
                    # Calculate returns
                    returns_df = df.pct_change().dropna()
                    
                    if not returns_df.empty and market_ticker in returns_df.columns:
                        market_returns = returns_df[market_ticker]
                        stock_returns = returns_df.drop(columns=[market_ticker])
                        
                        # Calculate statistics
                        stats_dict = calculate_statistics(returns_df, market_ticker)
                        
                        # Calculate historical market return
                        market_return_annual = market_returns.mean() * 252
                        
                        # Display market return assumptions
                        st.markdown("#### 📊 Market Return Assumptions")
                        col_hist, col_exp = st.columns(2)
                        with col_hist:
                            st.metric(
                                "Historical Market Return",
                                f"{market_return_annual:.2%}",
                                delta=None,
                                help="Actual market return during the selected period (used for beta calculation)"
                            )
                        with col_exp:
                            st.metric(
                                "Expected Future Market Return",
                                f"{expected_market_return:.2%}",
                                delta=None,
                                help="Assumed future market return for CAPM calculations"
                            )
                        
                        with tab2:
                            st.markdown('<h3 class="sub-header">CAPM Results</h3>', unsafe_allow_html=True)
                            
                            # Create results dataframe using EXPECTED market return
                            results = []
                            for ticker, stats in stats_dict.items():
                                expected_return = risk_free_rate + stats['Beta'] * (expected_market_return - risk_free_rate)
                                results.append({
                                    'Ticker': ticker,
                                    'Beta': stats['Beta'],
                                    'Alpha (daily)': stats['Alpha'],
                                    'Expected Return (annual)': expected_return,
                                    'R-squared': stats['R-squared'],
                                    'Sharpe Ratio': stats['Sharpe Ratio']
                                })
                            
                            results_df = pd.DataFrame(results)
                            
                            # Display in two columns
                            col_left, col_right = st.columns(2)
                            
                            with col_left:
                                st.markdown("#### 📊 Key Metrics")
                                # Use custom styling instead of background_gradient
                                styled_df = results_df.style.format({
                                    'Beta': '{:.3f}',
                                    'Alpha (daily)': '{:.6f}',
                                    'Expected Return (annual)': '{:.2%}',
                                    'R-squared': '{:.3f}',
                                    'Sharpe Ratio': '{:.3f}'
                                }).applymap(color_beta, subset=['Beta']) \
                                  .applymap(lambda x: color_return(x, expected_market_return), 
                                           subset=['Expected Return (annual)'])
                                
                                st.dataframe(styled_df, use_container_width=True)
                                
                                # Export CAPM results
                                st.markdown("---")
                                csv_capm = convert_df_to_csv(results_df)
                                st.download_button(
                                    label="📥 Download CAPM Results",
                                    data=csv_capm,
                                    file_name=f"capm_results_{start_date}_{end_date}.csv",
                                    mime="text/csv"
                                )
                            
                            with col_right:
                                st.markdown("#### 📈 Beta Distribution")
                                fig_beta = go.Figure(data=[
                                    go.Bar(
                                        x=results_df['Ticker'],
                                        y=results_df['Beta'],
                                        text=results_df['Beta'].round(3),
                                        textposition='auto',
                                        marker_color=['#3B82F6' if b < 1 else '#10B981' if b == 1 else '#EF4444' 
                                                    for b in results_df['Beta']]
                                    )
                                ])
                                fig_beta.add_hline(y=1, line_dash="dash", line_color="gray", 
                                                 annotation_text="Market Beta = 1")
                                fig_beta.update_layout(
                                    title="Stock Betas vs Market",
                                    xaxis_title="Ticker",
                                    yaxis_title="Beta",
                                    height=400,
                                    template='plotly_white'
                                )
                                st.plotly_chart(fig_beta, use_container_width=True)
                                
                                # Beta interpretation
                                st.markdown("""
                                <div class='info-box'>
                                <strong>Beta Interpretation:</strong><br>
                                • <span style='color:#EF4444'>Beta > 1</span>: More volatile than market (Aggressive)<br>
                                • <span style='color:#10B981'>Beta = 1</span>: Moves with market<br>
                                • <span style='color:#3B82F6'>Beta < 1</span>: Less volatile than market (Defensive)
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with tab3:
                            st.markdown('<h3 class="sub-header">Portfolio Analysis</h3>', unsafe_allow_html=True)
                            
                            if len(stats_dict) > 1:
                                # Calculate portfolio metrics
                                if custom_weights:
                                    # Use user-defined weights
                                    portfolio_beta = sum(stats_dict[ticker]['Beta'] * weights.get(ticker, 0) 
                                                       for ticker in stats_dict.keys())
                                    portfolio_alpha = sum(stats_dict[ticker]['Alpha'] * weights.get(ticker, 0) 
                                                        for ticker in stats_dict.keys())
                                    portfolio_sharpe = sum(stats_dict[ticker]['Sharpe Ratio'] * weights.get(ticker, 0) 
                                                         for ticker in stats_dict.keys())
                                    portfolio_vol = np.sqrt(sum(
                                        (weights.get(ticker, 0) ** 2) * (stats_dict[ticker]['Std Dev (daily)'] ** 2)
                                        for ticker in stats_dict.keys()
                                    )) * np.sqrt(252)
                                else:
                                    # Equal weights
                                    n = len(stats_dict)
                                    portfolio_beta = np.mean([stats['Beta'] for stats in stats_dict.values()])
                                    portfolio_alpha = np.mean([stats['Alpha'] for stats in stats_dict.values()])
                                    portfolio_sharpe = np.mean([stats['Sharpe Ratio'] for stats in stats_dict.values()])
                                    portfolio_vol = np.mean([stats['Volatility (annual)'] for stats in stats_dict.values()])
                                
                                # Use EXPECTED market return for portfolio return calculation
                                portfolio_return = risk_free_rate + portfolio_beta * (expected_market_return - risk_free_rate)
                                
                                # Display portfolio metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Portfolio Beta", f"{portfolio_beta:.3f}")
                                with col2:
                                    st.metric("Expected Return", f"{portfolio_return:.2%}")
                                with col3:
                                    st.metric("Sharpe Ratio", f"{portfolio_sharpe:.3f}")
                                with col4:
                                    st.metric("Annual Volatility", f"{portfolio_vol:.2%}")
                                
                                # Show CAPM formula with values
                                st.markdown("#### 🧮 CAPM Formula Applied")
                                st.latex(r'ER_p = R_f + \beta_p \times (ER_m - R_f)')
                                st.write(f"**Where:**")
                                st.write(f"- **R_f** = Risk-Free Rate = {risk_free_rate:.2%}")
                                st.write(f"- **ER_m** = Expected Market Return = {expected_market_return:.2%}")
                                st.write(f"- **β_p** = Portfolio Beta = {portfolio_beta:.3f}")
                                st.write(f"- **ER_p** = Expected Portfolio Return = {portfolio_return:.2%}")
                                
                                # Portfolio vs Market comparison
                                fig_portfolio = go.Figure()
                                fig_portfolio.add_trace(go.Bar(
                                    name='Portfolio',
                                    x=['Beta', 'Expected Return', 'Sharpe Ratio'],
                                    y=[portfolio_beta, portfolio_return, portfolio_sharpe],
                                    marker_color='#3B82F6'
                                ))
                                fig_portfolio.add_trace(go.Bar(
                                    name='Market',
                                    x=['Beta', 'Expected Return', 'Sharpe Ratio'],
                                    y=[1.0, expected_market_return, market_returns.mean()/market_returns.std() * np.sqrt(252)],
                                    marker_color='#6B7280'
                                ))
                                
                                fig_portfolio.update_layout(
                                    title="Portfolio vs Market Comparison",
                                    barmode='group',
                                    height=400,
                                    template='plotly_white'
                                )
                                st.plotly_chart(fig_portfolio, use_container_width=True)
                                
                                # Weight visualization
                                if custom_weights:
                                    st.markdown("#### 🎯 Portfolio Weights")
                                    fig_weights = go.Figure(data=[go.Pie(
                                        labels=list(weights.keys()),
                                        values=list(weights.values()),
                                        hole=0.3,
                                        marker_colors=px.colors.qualitative.Set3
                                    )])
                                    fig_weights.update_layout(title="Portfolio Allocation")
                                    st.plotly_chart(fig_weights, use_container_width=True)
                                    
                                    # Export portfolio allocation
                                    weights_df = pd.DataFrame({
                                        'Ticker': list(weights.keys()),
                                        'Weight': [f"{w*100:.1f}%" for w in weights.values()],
                                        'Allocation': list(weights.values())
                                    })
                                    csv_weights = convert_df_to_csv(weights_df)
                                    st.download_button(
                                        label="📥 Download Portfolio Allocation",
                                        data=csv_weights,
                                        file_name=f"portfolio_allocation_{start_date}_{end_date}.csv",
                                        mime="text/csv"
                                    )
                            else:
                                st.info("Add more stocks for portfolio analysis.")
                        
                        with tab4:
                            st.markdown('<h3 class="sub-header">Risk Metrics</h3>', unsafe_allow_html=True)
                            
                            # Risk comparison chart
                            risk_metrics = []
                            for ticker, stats in stats_dict.items():
                                risk_metrics.append({
                                    'Ticker': ticker,
                                    'Beta': stats['Beta'],
                                    'Volatility': stats['Volatility (annual)'],
                                    'Max Drawdown': abs(stats['Max Drawdown']),
                                    'Sharpe Ratio': stats['Sharpe Ratio']
                                })
                            
                            risk_df = pd.DataFrame(risk_metrics)
                            
                            # Scatter plot: Risk vs Return
                            fig_risk_return = px.scatter(
                                risk_df,
                                x='Volatility',
                                y='Sharpe Ratio',
                                size='Beta',
                                color='Ticker',
                                hover_name='Ticker',
                                title="Risk-Return Profile: Volatility vs Sharpe Ratio",
                                labels={'Volatility': 'Annual Volatility', 'Sharpe Ratio': 'Sharpe Ratio'}
                            )
                            fig_risk_return.update_layout(height=500)
                            st.plotly_chart(fig_risk_return, use_container_width=True)
                            
                            # Export risk metrics
                            csv_risk = convert_df_to_csv(risk_df)
                            st.download_button(
                                label="📥 Download Risk Metrics",
                                data=csv_risk,
                                file_name=f"risk_metrics_{start_date}_{end_date}.csv",
                                mime="text/csv"
                            )
                            
                            # Drawdown visualization
                            st.markdown("#### 📉 Maximum Drawdown Analysis")
                            for ticker in stock_returns.columns:
                                if ticker in stats_dict:
                                    cumulative = (1 + stock_returns[ticker]).cumprod()
                                    running_max = cumulative.expanding().max()
                                    drawdown = (cumulative - running_max) / running_max
                                    
                                    fig_dd = go.Figure()
                                    fig_dd.add_trace(go.Scatter(
                                        x=drawdown.index,
                                        y=drawdown,
                                        fill='tozeroy',
                                        fillcolor='rgba(239, 68, 68, 0.3)',
                                        line_color='#EF4444',
                                        name='Drawdown'
                                    ))
                                    fig_dd.update_layout(
                                        title=f"{ticker} - Maximum Drawdown: {abs(stats_dict[ticker]['Max Drawdown']):.2%}",
                                        xaxis_title="Date",
                                        yaxis_title="Drawdown",
                                        yaxis_tickformat='.1%',
                                        height=300,
                                        showlegend=False,
                                        template='plotly_white'
                                    )
                                    st.plotly_chart(fig_dd, use_container_width=True)
                    
                    else:
                        st.error(f"Market data for '{market_ticker}' not found.")
                
                else:
                    st.error("❌ No data returned. Please check your inputs.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
    else:
        st.warning("Please enter at least one ticker symbol.")

# Information section
with st.expander("📚 About CAPM & How to Use", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 How to Use This App
        
        1. **Enter Stock Tickers** - Use comma-separated symbols
        2. **Set Date Range** - Choose analysis period
        3. **Adjust Parameters** - Risk-free rate, expected market return
        4. **Set Portfolio Weights** - Customize allocation
        5. **Click Calculate** - Run comprehensive analysis
        
        ### 📥 Data Export Features
        
        • **Raw Price Data** - Download extracted prices as CSV
        • **Returns Data** - Download daily returns
        • **Normalized Data** - Download base-1 normalized prices
        • **CAPM Results** - Download calculated metrics
        • **Portfolio Allocation** - Download weight distribution
        • **Risk Metrics** - Download risk analysis
        """)
    
    with col2:
        st.markdown("""
        ### 📈 CAPM Formula
        
        $$
        ER_i = R_f + \\beta_i \\times (ER_m - R_f)
        $$
        
        Where:
        - $ER_i$ = Expected return of stock i
        - $R_f$ = Risk-free rate
        - $\\beta_i$ = Beta of stock i
        - $ER_m$ = Expected market return
        
        ### 💡 Investment Insights
        
        • **High Beta (>1.5)**: High risk, high potential return
        • **Low Beta (<0.5)**: Defensive, stable returns
        • **Negative Alpha**: Underperforming expectations
        • **Positive Alpha**: Outperforming expectations
        • **Market Return Assumption**: Use 7% for long-term S&P 500 average
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 1rem;'>
    <strong>CAPM Calculator Pro</strong> | For Educational Purposes Only | Data Source: Yahoo Finance
</div>
""", unsafe_allow_html=True)
