import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

# 🌍 Load .env for Bybit API keys
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# 🔑 Start Bybit session
session = HTTP(
    api_key=API_KEY,
    api_secret=API_SECRET,
)

st.set_page_config(page_title="Crypto Capital Dashboard", layout="wide")

# 📂 Path to your trades.csv
CSV_FILE = "trades.csv"

# 🔄 Load bot trades
@st.cache_data(ttl=15)
def load_trades():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        return df
    else:
        return pd.DataFrame()

# 🔄 Load live open futures positions
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

# 🧠 Load data
df_trades = load_trades()
df_open_positions = load_open_positions()

# 🚀 DASHBOARD UI
st.title("🚀 Crypto Capital Trading Dashboard")
st.markdown("---")

# 📈 Bot Trades Log
st.subheader("📈 Bot Trades Log (From trades.csv)")

if df_trades.empty:
    st.warning("No bot trades found yet. Waiting for first trade...")
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

    # 📊 Basic Stats
    st.markdown("---")
    st.subheader("📊 Bot Trading Stats")
    total_trades = len(df_trades)
    total_qty = df_trades['qty'].sum()

    if entry_col:
        avg_entry = df_trades[entry_col].mean()
    else:
        avg_entry = 0

    st.metric(label="Total Trades", value=total_trades)
    st.metric(label="Total Quantity Traded", value=f"{total_qty:.3f}")
    st.metric(label="Average Entry Price", value=f"${avg_entry:.2f}")

# 📊 Open Manual Positions Section
st.markdown("---")
st.subheader("🔎 Live Open Futures Positions (Manual & Bot)")

if df_open_positions.empty:
    st.warning("No open futures positions found.")
else:
    # 📈 Add PnL formatting and coloring
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

# 📥 Place Live Futures Trade
st.markdown("---")
st.header("🛒 Place Live Futures Trade")

st.subheader("⚡ Quick Trade Panel")

# Dropdown for symbol
symbol_choice = st.selectbox(
    "Select Symbol", 
    ("BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT")
)

# Dropdown for side
side_choice = st.selectbox(
    "Side",
    ("Buy", "Sell")
)

# Input for quantity
quantity = st.number_input(
    "Quantity (Contracts)", 
    min_value=0.001, 
    step=0.001, 
    format="%.3f"
)

# Button to submit
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

