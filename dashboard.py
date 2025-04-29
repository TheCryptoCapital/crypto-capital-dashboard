import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

# 🌍 Load API keys
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# 📈 Connect to Bybit
session = HTTP(api_key=API_KEY, api_secret=API_SECRET)

# 📂 Load Bot Trades CSV
CSV_FILE = "trades.csv"

st.set_page_config(page_title="The Crypto Capital", layout="wide")

# 🖌️ Custom CSS: Bigger font, normal, spaced, white tabs
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 20px;
        font-weight: normal;
        margin-right: 20px;
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Load Bot Trades
@st.cache_data(ttl=15)
def load_trades():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        return pd.DataFrame()

# 🧠 Load Open Manual Futures Positions
@st.cache_data(ttl=15)
def load_open_positions():
    try:
        response = session.get_positions(category="linear", settleCoin="USDT", accountType="UNIFIED")
        futures_data = response["result"]["list"]

        open_positions = []
        for pos in futures_data:
            size = float(pos.get("size", 0))
            if size != 0:
                open_positions.append({
                    "Symbol": pos.get("symbol", ""),
                    "Size": size,
                    "Entry Price": float(pos.get("avgPrice", 0)),
                    "Mark Price": float(pos.get("markPrice", 0)),
                    "PnL ($)": float(pos.get("unrealisedPnl", 0)),
                    "Leverage": pos.get("leverage", "N/A")
                })
        return pd.DataFrame(open_positions)
    except Exception as e:
        print(f"⚠️ Error loading open futures positions: {e}")
        return pd.DataFrame()

# 🧠 Build Growth Curve
def build_growth_curve(df_trades):
    if df_trades.empty:
        return None
    df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
    df_trades = df_trades.sort_values('timestamp')
    df_trades['Cumulative PnL'] = df_trades['take_profit'] - df_trades['stop_loss']
    df_trades['Cumulative PnL'] = df_trades['Cumulative PnL'].cumsum()
    fig = px.line(df_trades, x="timestamp", y="Cumulative PnL", title="Growth Curve (PnL Over Time)")
    return fig

# 🧠 Split Bot Trades into Open and Closed
def split_bot_trades(df_trades):
    if df_trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    if 'take_profit' in df_trades.columns and 'stop_loss' in df_trades.columns:
        df_open = df_trades[df_trades['take_profit'] == 0]
        df_closed = df_trades[df_trades['take_profit'] != 0]
        return df_open, df_closed
    return df_trades, pd.DataFrame()

# 🧠 Load Data
df_trades = load_trades()
df_open_positions = load_open_positions()
df_bot_open, df_bot_closed = split_bot_trades(df_trades)

# 🚀 Dashboard Title
st.title("🚀 The Crypto Capital Dashboard")

# 📊 TABS (All Trades First)
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌐 All Trades", "📈 Bot Open Trades", "✅ Bot Closed Trades",
    "🔥 Manual Open Trades", "✅ Manual Closed Trades",
    "📊 Growth Curve", "🛒 Place Trade"
])

# 🟠 TAB 1: All Trades
with tab1:
    st.subheader("🌐 All Trades Summary View")
    if df_trades.empty and df_open_positions.empty:
        st.warning("No trades yet.")
    else:
        if not df_bot_closed.empty:
            st.subheader("✅ Completed Bot Trades")
            st.dataframe(df_bot_closed)
        if not df_open_positions.empty:
            st.subheader("🔥 Live Manual Open Trades")
            st.dataframe(df_open_positions)

# 🟠 TAB 2: Bot Open Trades
with tab2:
    st.subheader("📈 Active Bot Trades")
    if df_bot_open.empty:
        st.info("No active bot trades.")
    else:
        st.dataframe(df_bot_open)

# 🟠 TAB 3: Bot Closed Trades
with tab3:
    st.subheader("✅ Completed Bot Trades")
    if df_bot_closed.empty:
        st.info("No closed bot trades yet.")
    else:
        st.dataframe(df_bot_closed)

# 🟠 TAB 4: Manual Open Trades
with tab4:
    st.subheader("🔥 Live Manual Open Trades")
    if df_open_positions.empty:
        st.warning("No live manual trades open.")
    else:
        def color_pnl(val):
            try:
                val = float(val)
                if val > 0:
                    return 'color: green;'
                elif val < 0:
                    return 'color: red;'
                else:
                    return ''
            except:
                return ''
        st.dataframe(
            df_open_positions.style.format({
                'Entry Price': '{:.6f}',
                'Mark Price': '{:.6f}',
                'PnL ($)': '{:+.2f}',
                'Size': '{:.3f}',
                'Leverage': '{}'
            }).applymap(color_pnl, subset=['PnL ($)'])
        )

# 🟠 TAB 5: Manual Closed Trades (Coming Soon)
with tab5:
    st.subheader("✅ Completed Manual Trades (Coming Soon)")
    st.info("Manual closed trades tracking will be added later.")

# 🟠 TAB 6: Growth Curve
with tab6:
    st.subheader("📊 Trading Growth Curve")
    if df_trades.empty:
        st.warning("No bot trades yet to plot growth.")
    else:
        fig = build_growth_curve(df_trades)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# 🟠 TAB 7: Place Trade
with tab7:
    st.subheader("🛒 Place a Live Futures Trade")

    symbol_choice = st.selectbox(
        "Select Symbol", 
        ("BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT")
    )

    side_choice = st.selectbox(
        "Side",
        ("Buy", "Sell")
    )

    quantity = st.number_input(
        "Quantity (Contracts)", 
        min_value=0.001, 
        step=0.001, 
        format="%.3f"
    )

    if st.button("🚀 Place Market Order"):
        try:
            order = session.place_order(
                category="linear",
                symbol=symbol_choice,
                side=side_choice,
                orderType="Market",
                qty=round(quantity, 3),
                timeInForce="GoodTillCancel",
                reduceOnly=False,
                closeOnTrigger=False
            )
            st.success(f"✅ Order placed: {side_choice} {quantity} {symbol_choice}")
        except Exception as e:
            st.error(f"❌ Order failed: {e}")

st.caption("🔄 Auto-refresh every 15 seconds...")
