import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pro Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to tighten up the UI
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS & CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def generate_mock_data(ticker, days):
    """
    Generates synthetic financial time-series data to simulate an API call.
    In a real app, replace this with yfinance or an API request.
    """
    date_rng = pd.date_range(end=datetime.today(), periods=days)
    
    # Random walk generation
    base_price = 50000 if ticker == 'BTC-USD' else 150
    volatility = 0.02
    
    prices = []
    price = base_price
    for _ in range(days):
        change = np.random.normal(0, volatility)
        price = price * (1 + change)
        prices.append(price)
        
    df = pd.DataFrame(data={'Date': date_rng, 'Close': prices})
    df['Open'] = df['Close'] * (1 + np.random.normal(0, 0.005, days))
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.01, days)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.01, days)))
    df['Volume'] = np.random.randint(1000, 100000, days)
    df.set_index('Date', inplace=True)
    
    return df

def create_candlestick_chart(df, ma_window):
    """Creates a professional Plotly candlestick chart"""
    df['MA'] = df['Close'].rolling(window=ma_window).mean()
    
    fig = go.Figure()
    
    # Candlestick trace
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='Market Data'
    ))
    
    # Moving Average trace
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA'],
        mode='lines', name=f'{ma_window}-Day MA',
        line=dict(color='orange', width=1.5)
    ))
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_white",
        xaxis_rangeslider_visible=False
    )
    return fig

# -----------------------------------------------------------------------------
# 3. SIDEBAR UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Data Source Selection
    ticker = st.selectbox(
        "Select Asset",
        ["BTC-USD", "ETH-USD", "AAPL", "GOOGL", "TSLA"],
        index=0
    )
    
    # Time Range Selection
    time_range = st.radio(
        "Time Range",
        ["30 Days", "90 Days", "1 Year"],
        horizontal=True
    )
    
    days_map = {"30 Days": 30, "90 Days": 90, "1 Year": 365}
    days_selected = days_map[time_range]
    
    st.divider()
    
    # Analysis Parameters
    st.subheader("Analysis Parameters")
    ma_window = st.slider("Moving Average Window", 5, 100, 20)
    show_volume = st.toggle("Show Volume Data", value=True)
    
    st.divider()
    
    # Action Button
    if st.button("Reset Analysis", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD UI
# -----------------------------------------------------------------------------

# Load Data (simulated delay for realism)
with st.spinner(f"Fetching data for {ticker}..."):
    df = generate_mock_data(ticker, days_selected)

# --- KPI Row ---
current_price = df['Close'].iloc[-1]
prev_price = df['Close'].iloc[-2]
delta = ((current_price - prev_price) / prev_price) * 100
avg_vol = df['Volume'].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Current Price", f"${current_price:,.2f}", f"{delta:.2f}%")
with col2:
    st.metric("Highest Price (Period)", f"${df['High'].max():,.2f}")
with col3:
    st.metric("Lowest Price (Period)", f"${df['Low'].min():,.2f}")
with col4:
    st.metric("Avg Volume", f"{avg_vol:,.0f}")

st.markdown("---")

# --- Tabbed Layout ---
tab1, tab2, tab3 = st.tabs(["📈 Chart", "🧮 Data", "🤖 Forecasting"])

with tab1:
    st.subheader(f"{ticker} Price Trends")
    
    # Interactive Plotly Chart
    fig = create_candlestick_chart(df, ma_window)
    st.plotly_chart(fig, use_container_width=True)
    
    if show_volume:
        st.subheader("Volume Analysis")
        st.bar_chart(df['Volume'], color="#7792E3", height=200)

with tab2:
    col_left, col_right = st.columns([3, 1])
    
    with col_left:
        st.subheader("Raw Dataset")
        st.dataframe(df.style.highlight_max(axis=0), use_container_width=True, height=400)
    
    with col_right:
        st.subheader("Export")
        st.write("Download this dataset for external analysis.")
        csv = df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f'{ticker}_data.csv',
            mime='text/csv',
            use_container_width=True
        )
        
        st.subheader("Stats")
        st.write(df.describe())

with tab3:
    st.info("This section connects to the ARIMA model defined in your notebook.")
    
    c1, c2 = st.columns(2)
    with c1:
        forecast_days = st.number_input("Days to Forecast", min_value=1, max_value=30, value=7)
    with c2:
        model_type = st.selectbox("Model Type", ["ARIMA", "SARIMA", "Prophet"])
        
    if st.button("Run Forecast"):
        st.success(f"Generating {model_type} forecast for next {forecast_days} days...")
        
        # Simulation of forecast data
        future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_days)
        future_prices = [current_price * (1 + np.random.normal(0, 0.01)) for _ in range(forecast_days)]
        forecast_df = pd.DataFrame({'Forecast': future_prices}, index=future_dates)
        
        st.line_chart(forecast_df)

# -----------------------------------------------------------------------------
# 5. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(f"Dashboard v1.0 | Data last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")