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

# 🧠 Load bot trades
@st.cache_data(ttl=15)
def load_trades():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        return pd.DataFrame()

# 🧠 Load manual open futures positions
@st.cache_data(ttl=15)
def load_open_positions():
    try:
        response = session.get_positions(category="linear", settleCoin="USDT")
        futures_data = response["result"]["list"]

        open_positions = []
        for pos in futures_data:
            position_value = float(pos.get("positionValue", 0))
            avg_price = float(pos.get("avgPrice", 0))
            mark_price = float(pos.get("markPrice", 0))
            unrealised_pnl = float(pos.get("unrealisedPnL", 0))
            leverage = pos.get("leverage", "N/A")
            symbol = pos.get("symbol", "N/A")

            if position_value != 0:
                open_positions.append({
                    "Symbol": symbol,
                    "Avg Entry Price": avg_price,
                    "Mark Price": mark_price,
                    "Position Value ($)": position_value,
                    "Unrealized PnL ($)": unrealised_pnl,
                    "Leverage": leverage
                })

        return pd.DataFrame(open_positions)
    except Exception as e:
        print(f"⚠️ Error loading futures positions: {e}")
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

# 🧠 Load Data
df_trades = load_trades()
df_open_positions = load_open_positions()

# 🚀 Dashboard Title
st.title("🚀 The Crypto Capital")

# 📊 TABS
tab1, tab2, tab3, tab4 = st.tabs(["📈 Bot Trades", "🔥 Manual Open Trades", "📊 Growth Curve", "🛒 Place Trade"])

# 🟠 TAB 1: Bot Trades
with tab1:
    st.subheader("📈 Bot Trades Log (From trades.csv)")
    if df_trades.empty:
        st.warning("No bot trades found yet.")
    else:
        if 'entry_price' in df_trades.columns:
            entry_col = 'entry_price'
        elif 'entry' in df_trades.columns:
            entry_col = 'entry'
        else:
            entry_col = None

        if entry_col:
            st.dataframe(df_trades.style.format({
                entry_col: '{:.2f}',
                'stop_loss': '{:.2f}',
                'take_profit': '{:.2f}',
                'qty': '{:.3f}'
            }))
        else:
            st.dataframe(df_trades)

# 🟠 TAB 2: Manual Open Trades
with tab2:
    st.subheader("🔥 Live Manual Open Futures Positions")
    if df_open_positions.empty:
        st.warning("No open manual futures positions found.")
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
                'Avg Entry Price': '{:.6f}',
                'Mark Price': '{:.6f}',
                'Position Value ($)': '{:.2f}',
                'Unrealized PnL ($)': '{:+.2f}',
                'Leverage': '{}'
            }).applymap(color_pnl, subset=['Unrealized PnL ($)'])
        )

# 🟠 TAB 3: Growth Curve
with tab3:
    st.subheader("📊 Bot Trading Growth Curve (PnL Over Time)")
    if df_trades.empty:
        st.warning("No bot trades yet to generate growth curve.")
    else:
        fig = build_growth_curve(df_trades)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# 🟠 TAB 4: Place Trade Form
with tab4:
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

