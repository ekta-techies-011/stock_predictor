import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objs as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Stock Market Price Prediction Dashboard",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Main Title
st.title("📈 Stock Market Price Prediction Dashboard")
st.markdown("🔮 Advanced AI-Powered Stock Analysis & Prediction Platform")

# Services Description
with st.expander("🚀 Our Services", expanded=False):
    st.markdown("""
    **📊 Comprehensive Stock Analysis:**
    - Real-time data collection from Yahoo Finance
    - Advanced technical indicators (RSI, MACD, Moving Averages)
    - Volatility analysis and trend detection
    
    **🤖 AI-Powered Predictions:**
    - Machine Learning models for price prediction
    - Classification models for buy/sell signals
    - Ensemble methods for improved accuracy
    
    **📈 Interactive Visualizations:**
    - Interactive charts with zoom and hover capabilities
    - Real-time data updates
    - Professional-grade financial charts
    
    **🎯 Investment Insights:**
    - Buy/sell signal generation
    - Risk assessment and volatility analysis
    - Performance metrics and model evaluation
    """)

# Initialize data variable
data = None

# Sidebar
st.sidebar.markdown("## 🎛️ Dashboard Controls")
st.sidebar.markdown("---")

# Data source selection
source = st.sidebar.radio("📥 Data Source", ["Download from Yahoo Finance", "Use Local CSV"], index=0)

# Ticker and date input
if source == "Download from Yahoo Finance":
    st.sidebar.markdown("### 📊 Download New Data")
    ticker = st.sidebar.text_input("Stock Ticker (e.g. AAPL, TCS.NS)", value="MSFT").upper()
    start_date = st.sidebar.date_input("Start Date:", value=datetime(2010,1,1))
    end_date = st.sidebar.date_input("End Date:", value=datetime(2025,1,1))
    if st.sidebar.button("🚀 Download Data", type="primary"):
        with st.spinner("Downloading data from Yahoo Finance..."):
            data = yf.download(ticker, start=start_date, end=end_date)
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])
            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                if all(isinstance(col, tuple) and len(col) == 2 for col in data.columns):
                    data.columns = [col[0] for col in data.columns]
                else:
                    data.columns = ['_'.join([str(i) for i in col if i]) for col in data.columns.values]
            # Standardize column names
            data.columns = [str(c).strip().capitalize() for c in data.columns]
            if not data.empty:
                data.to_csv(f"data/{ticker}_data.csv", index=False)
                st.sidebar.success(f"✅ Data for {ticker} saved to data/{ticker}_data.csv")
            else:
                st.sidebar.error("❌ No data found for the specified ticker and date range.")
    else:
        st.sidebar.info("📊 Click 'Download Data' to load stock data for analysis")
else:
    st.sidebar.markdown("### 📁 Use Local Data")
    csv_files = [f for f in os.listdir("data") if f.endswith("_data.csv")]
    if not csv_files:
        st.sidebar.warning("⚠️ No local CSV files found in /data. Please download data first.")
        st.stop()
    selected_csv = st.sidebar.selectbox("Select Local CSV", csv_files)
    ticker = selected_csv.split("_data.csv")[0]
    data = pd.read_csv(f"data/{selected_csv}")
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        if all(isinstance(col, tuple) and len(col) == 2 for col in data.columns):
            data.columns = [col[0] for col in data.columns]
        else:
            data.columns = ['_'.join([str(i) for i in col if i]) for col in data.columns.values]
    # Standardize column names
    data.columns = [str(c).strip().capitalize() for c in data.columns]
    data['Date'] = pd.to_datetime(data['Date'])
    st.sidebar.success(f"✅ Loaded {selected_csv}")

# Ensure data is loaded before proceeding
if data is None or data.empty:
    st.warning("⚠️ Please load data first using the sidebar.")
    st.stop()

# Data Cleaning
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

# Feature Engineering
price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in price_columns:
    if col in data.columns:
        if isinstance(data[col], pd.Series):
            data[col] = pd.to_numeric(data[col], errors='coerce')
        else:
            st.warning(f"⚠️ Column '{col}' is not a Series. It may be duplicated or malformed. Skipping.")
    else:
        st.warning(f"⚠️ Column '{col}' not found in data. Some features/plots may not work.")

data.dropna(inplace=True)
data['Daily_Return'] = data['Close'].pct_change()
data['MA_7'] = data['Close'].rolling(window=7).mean()
data['MA_21'] = data['Close'].rolling(window=21).mean()
data['Volatility_7'] = data['Daily_Return'].rolling(window=7).std()
data['Prev_Close'] = data['Close'].shift(1)

# RSI
window = 14
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
rs = gain / loss
data['RSI_14'] = 100 - (100 / (1 + rs))

# MACD
ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26
data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
data.dropna(inplace=True)

# Tabs for Navigation
tabs = st.tabs(["📊 Overview", "📈 Visualizations", "🤖 Regression", "🎯 Classification", "📡 Buy/Sell Signals"])

# Overview Tab
with tabs[0]:
    st.header("📊 Data Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", f"{len(data):,}")
    with col2:
        st.metric("Latest Close", f"${data['Close'].iloc[-1]:.2f}")
    with col3:
        st.metric("Start Date", data['Date'].min().strftime('%Y-%m-%d'))
    with col4:
        st.metric("End Date", data['Date'].max().strftime('%Y-%m-%d'))
    
    # Data Preview
    st.subheader("📋 Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Correlation Heatmap
    st.subheader("🔗 Feature Correlation Matrix")
    corr = data[['Close', 'Daily_Return', 'MA_7', 'MA_21', 'Volatility_7', 'Prev_Close', 'RSI_14', 'MACD', 'MACD_Signal']].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', aspect="auto")
    fig.update_layout(title="Feature Correlation Heatmap", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

# Visualizations Tab
with tabs[1]:
    st.header("📈 Interactive Visualizations")
    
    # Price and Moving Averages
    st.subheader("📊 Price & Moving Averages (Last 2 Years)")
    recent_data = data[data['Date'] >= pd.to_datetime('2022-01-01')]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent_data['Date'], y=recent_data['Close'], mode='lines', name='Close Price', line=dict(width=3, color='#667eea')))
    fig.add_trace(go.Scatter(x=recent_data['Date'], y=recent_data['MA_7'], mode='lines', name='7-Day MA', line=dict(dash='dash', color='#764ba2')))
    fig.add_trace(go.Scatter(x=recent_data['Date'], y=recent_data['MA_21'], mode='lines', name='21-Day MA', line=dict(dash='dot', color='#f093fb')))
    fig.update_layout(xaxis_title='Date', yaxis_title='Price ($)', legend_title='Legend', template='plotly_white', 
                     title="Stock Price with Moving Averages", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    
    # Daily Returns Distribution
    st.subheader("📊 Daily Returns Distribution")
    fig = px.histogram(data, x='Daily_Return', nbins=80, marginal='box', color_discrete_sequence=['#667eea'], opacity=0.8)
    mean = data['Daily_Return'].mean()
    std = data['Daily_Return'].std()
    fig.add_vline(x=mean, line_dash='dash', line_color='green', annotation_text='Mean')
    fig.add_vline(x=mean+std, line_dash='dash', line_color='red', annotation_text='+1 Std')
    fig.add_vline(x=mean-std, line_dash='dash', line_color='red', annotation_text='-1 Std')
    fig.update_layout(title="Daily Returns Distribution", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # Volatility
    st.subheader("📊 7-Day Rolling Volatility")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Volatility_7'], mode='lines', name='7-Day Volatility', line=dict(color='#fa709a')))
    avg_vol = data['Volatility_7'].mean()
    fig.add_hline(y=avg_vol, line_dash='dash', line_color='blue', annotation_text='Avg Volatility')
    fig.update_layout(xaxis_title='Date', yaxis_title='Volatility', template='plotly_white', 
                     title="7-Day Rolling Volatility", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # RSI
    st.subheader("📊 RSI (14-day)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI_14'], mode='lines', name='RSI 14', line=dict(color='#764ba2')))
    fig.add_hline(y=70, line_dash='dash', line_color='red', annotation_text='Overbought (70)')
    fig.add_hline(y=30, line_dash='dash', line_color='blue', annotation_text='Oversold (30)')
    fig.update_layout(xaxis_title='Date', yaxis_title='RSI', template='plotly_white', 
                     title="Relative Strength Index (14-day)", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # MACD
    st.subheader("📊 MACD Indicator")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], mode='lines', name='MACD Line', line=dict(color='#4facfe')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='#00f2fe')))
    fig.add_hline(y=0, line_dash='dash', line_color='green')
    fig.update_layout(xaxis_title='Date', yaxis_title='MACD', template='plotly_white', 
                     title="MACD (Moving Average Convergence Divergence)", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

# Regression Tab
with tabs[2]:
    st.header("🤖 Regression Models: Next-Day Price Prediction")
    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    features = ['Close', 'Prev_Close', 'MA_7', 'MA_21', 'Volatility_7', 'RSI_14', 'MACD', 'MACD_Signal']
    X = data[features]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    # Get latest predictions
    latest_features = data[features].iloc[-1:]
    next_day_pred_lr = model.predict(latest_features)[0]
    next_day_pred_rf = rf_model.predict(latest_features)[0]
    
    # Model Performance Metrics
    st.subheader("📊 Model Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Linear Regression**")
        st.metric("RMSE", f"${rmse:.4f}")
        st.metric("MAE", f"${mae:.4f}")
        st.metric("R² Score", f"{r2:.4f}")
    
    with col2:
        st.markdown("**Random Forest**")
        st.metric("RMSE", f"${rmse_rf:.4f}")
        st.metric("MAE", f"${mae_rf:.4f}")
        st.metric("R² Score", f"{r2_rf:.4f}")
    
    # Latest Predictions
    st.subheader("🎯 Latest Predictions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Latest Actual Close:** ${data['Close'].iloc[-1]:.2f}")
    
    with col2:
        st.success(f"**Linear Regression Prediction:** ${next_day_pred_lr:.2f}")
    
    with col3:
        st.warning(f"**Random Forest Prediction:** ${next_day_pred_rf:.2f}")
    
    # Actual vs Predicted Plots
    st.subheader("📈 Actual vs Predicted Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Linear Regression**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test, mode='lines', name='Actual', line=dict(color='#667eea')))
        fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted', line=dict(color='#764ba2')))
        fig.update_layout(xaxis_title='Index', yaxis_title='Price ($)', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Random Forest**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test, mode='lines', name='Actual', line=dict(color='#667eea')))
        fig.add_trace(go.Scatter(y=y_pred_rf, mode='lines', name='RF Predicted', line=dict(color='#f093fb')))
        fig.update_layout(xaxis_title='Index', yaxis_title='Price ($)', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# Classification Tab
with tabs[3]:
    st.header("🎯 Classification: Up/Down Movement Prediction")
    data['Signal'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(subset=features + ['Signal'], inplace=True)
    X_class = data[features]
    y_class = data['Signal']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)
    
    acc = accuracy_score(y_test_c, y_pred_c)
    prec = precision_score(y_test_c, y_pred_c)
    rec = recall_score(y_test_c, y_pred_c)
    f1 = f1_score(y_test_c, y_pred_c)
    
    # Classification Metrics
    st.subheader("📊 Classification Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{prec:.4f}")
    col3.metric("Recall", f"{rec:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")
    
    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test_c, y_pred_c)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', aspect="auto", 
                   x=['Down','Up'], y=['Down','Up'])
    fig.update_layout(xaxis_title='Predicted', yaxis_title='Actual', 
                     title="Confusion Matrix", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation
    st.info("""
    **📝 Classification Explanation:**
    - **Accuracy:** Overall correctness of predictions
    - **Precision:** When we predict "Up", how often are we correct?
    - **Recall:** Of all actual "Up" movements, how many did we catch?
    - **F1 Score:** Harmonic mean of precision and recall
    """)

# Signals Tab
with tabs[4]:
    st.header("📡 Buy/Sell Signal Visualization")
    
    # Prepare data for signal plot
    X_class = data[features].copy()
    y_class = data['Signal'].copy()
    X_class['Original_Index'] = X_class.index
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, shuffle=False)
    y_pred_c = clf.predict(X_test_c[features])
    X_test_c = X_test_c.reset_index(drop=True)
    X_test_c['Predicted_Signal'] = y_pred_c
    X_test_c['Actual_Signal'] = y_test_c.values
    X_test_c['Date'] = data.loc[X_test_c['Original_Index'], 'Date'].values
    X_test_c['Close'] = data.loc[X_test_c['Original_Index'], 'Close'].values
    
    # Signal Summary
    buy_signals = X_test_c[X_test_c['Predicted_Signal'] == 1]
    sell_signals = X_test_c[X_test_c['Predicted_Signal'] == 0]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Signals", len(X_test_c))
    with col2:
        st.metric("Buy Signals", len(buy_signals))
    with col3:
        st.metric("Sell Signals", len(sell_signals))
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_test_c['Date'], y=X_test_c['Close'], mode='lines', 
                            name='Stock Price', line=dict(color='#667eea', width=2)))
    fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], mode='markers', 
                            name='Predicted Buy (Up)', marker=dict(symbol='triangle-up', color='green', size=12)))
    fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], mode='markers', 
                            name='Predicted Sell (Down)', marker=dict(symbol='triangle-down', color='red', size=12)))
    fig.update_layout(xaxis_title='Date', yaxis_title='Stock Price ($)', template='plotly_white', 
                     title="Buy/Sell Signal Visualization", title_x=0.5, legend_title='Signal')
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("🟢 Green triangles: Buy signals | 🔴 Red triangles: Sell signals")

# Terms & Conditions
st.markdown("---")
st.markdown("""
## Terms & Conditions

**Disclaimer:** This Stock Market Price Prediction Dashboard is for educational and research purposes only. The predictions and analysis provided are based on historical data and machine learning models, and should not be considered as financial advice.

**Risk Warning:** Stock market investments carry inherent risks. Past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.

**Data Source:** Market data is sourced from Yahoo Finance and may be subject to delays or inaccuracies. We do not guarantee the accuracy, completeness, or timeliness of the information provided.

**Model Limitations:** Machine learning models are based on historical patterns and may not accurately predict future market movements. Market conditions can change rapidly, affecting model performance.

**Use at Your Own Risk:** By using this dashboard, you acknowledge that you understand the risks involved and agree to use this tool responsibly. We are not liable for any financial losses or decisions made based on the information provided.

**Technical Indicators:** RSI, MACD, Moving Averages, and other technical indicators are tools for analysis but should not be used in isolation for trading decisions.

*Last updated: December 22, 2024*
""")