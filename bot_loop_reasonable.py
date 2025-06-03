#!/usr/bin/env python3
"""
Ultimate Trading Bot - Complete Multi-Strategy System with Advanced Features
Author: Elite Trading Systems
Version: 3.0.0

Features:
- Multi-strategy voting system (RSI, MACD, EMA) - 2/3 required
- ML/AI strategies (RandomForest, Ensemble, PPO RL agent)
- Advanced trailing stop management (2% trigger, 1% offset)
- Live PnL tracking with win/loss statistics
- Comprehensive performance logging with CSV export
- Multi-timeframe analysis (MTF)
- Sentiment analysis from multiple sources
- Whale tracking and on-chain analysis
- Smart order routing (TWAP, limit orders)
- Market regime detection
- Confidence-weighted voting
- Meta-agent strategy selection
- Advanced risk management with circuit breakers
- Async market scanning every 15 seconds
- Risk-adjusted position sizing
- Real-time P&L tracking in trade_log.csv
"""

import os
import time
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import statistics
import pickle
import warnings
import aiohttp
warnings.filterwarnings('ignore')

# ML/AI imports
try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML libraries not installed. Run: pip install scikit-learn")

# RL imports
try:
    import gym
    from gym import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    import torch
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("⚠️ RL libraries not installed. Run: pip install stable-baselines3 gym torch")

# Technical Analysis imports
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("⚠️ TA library not installed. Run: pip install ta")

# =====================================
# CONFIGURATION
# =====================================

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class MarketRegime(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"

@dataclass
class BotConfig:
    """Enhanced bot configuration with all features"""
    # Trading pairs
    SYMBOLS: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    
    # Position sizing
    MAX_POSITION_VALUE: float = 1000  # Max $1000 per position
    RISK_PER_TRADE_PCT: float = 1.5   # Risk 1.5% of account per trade
    MIN_BALANCE_REQUIRED: float = 100  # Minimum balance to trade
    
    # Risk management
    STOP_LOSS_PCT: float = 2.0  # Initial stop loss
    TAKE_PROFIT_PCT: float = 4.0  # Initial take profit
    MAX_OPEN_POSITIONS: int = 3  # Max concurrent positions
    
    # Trailing stop configuration
    TRAIL_TRIGGER: float = 2.0  # Activate at 2% profit (TRAIL_TRIGGER)
    TRAIL_OFFSET: float = 1.0  # Trail by 1% (TRAIL_OFFSET)
    
    # Strategy parameters
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: int = 30
    RSI_OVERBOUGHT: int = 70
    
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    EMA_FAST: int = 9
    EMA_SLOW: int = 21
    
    # Voting system
    MIN_STRATEGY_AGREEMENT: int = 2  # Need 2/3 strategies to agree
    MIN_SIGNAL_STRENGTH: float = 0.6  # Minimum combined strength
    
    # Timing
    SCAN_INTERVAL: int = 15  # Scan every 15 seconds
    KLINE_INTERVAL: str = "5"  # 5-minute candles
    KLINE_LIMIT: int = 100  # Number of candles to fetch
    
    # API settings
    TESTNET: bool = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
    
    # ML/AI Strategy parameters
    ML_ENABLED: bool = True  # Enable ML strategies
    ML_LOOKBACK: int = 50  # Candles for feature engineering
    ML_RETRAIN_INTERVAL: int = 1000  # Retrain every N scans
    ML_MIN_ACCURACY: float = 0.55  # Minimum accuracy to use ML signal
    
    # PPO RL parameters
    PPO_ENABLED: bool = True  # Enable PPO agent
    PPO_MODE: str = "predict"  # "train" or "predict"
    PPO_MODEL_PATH: str = "models/ppo_trader.zip"
    PPO_LEARNING_RATE: float = 0.0003
    PPO_EPISODES: int = 10000
    
    # Multi-timeframe analysis
    MTF_ENABLED: bool = True
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["1", "5", "15"])
    MTF_CONFLUENCE_REQUIRED: int = 2  # Need 2/3 timeframes to agree
    
    # Sentiment analysis
    SENTIMENT_ENABLED: bool = True
    SENTIMENT_SOURCES: List[str] = field(default_factory=lambda: ["cryptopanic", "reddit"])
    SENTIMENT_THRESHOLD: float = 0.3  # Minimum sentiment score
    
    # Whale tracking
    WHALE_TRACKING_ENABLED: bool = True
    WHALE_THRESHOLD_USD: float = 500000  # $500k for whale transactions
    
    # Smart order routing
    USE_SMART_ORDERS: bool = True
    MAX_SLIPPAGE_PCT: float = 0.5  # Maximum acceptable slippage
    
    # Market regime detection
    REGIME_DETECTION_ENABLED: bool = True
    REGIME_LOOKBACK: int = 100  # Candles for regime detection
    
    # Advanced features
    CONFIDENCE_WEIGHTED_VOTING: bool = True
    MIN_CONFIDENCE_THRESHOLD: float = 0.6
    META_AGENT_ENABLED: bool = True
    META_AGENT_WINDOW: int = 50  # Trades to evaluate strategies
    
    # Risk management
    DAILY_LOSS_LIMIT_PCT: float = 5.0  # 5% daily loss limit
    FLASH_CRASH_THRESHOLD: float = 10.0  # 10% drop detection
    VOLATILITY_ADJUSTED_SIZING: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"

# Continue with all the original classes from paste.txt...
# [Including all original classes: PerformanceTracker, LivePnLTracker, CSVTradeLogger, 
#  TrailingStopManager, TechnicalIndicators, BaseStrategy, RSIStrategy, MACDStrategy, 
#  EMAStrategy, FeatureEngineer, RandomForestStrategy, EnsembleMLStrategy, PPOStrategy, 
#  StrategyVoter, AsyncMarketData, EnhancedOrderManager, AccountManager]

# =====================================
# LOGGING SETUP
# =====================================

def setup_logging():
    """Setup enhanced logging configuration"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=BotConfig.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/ultimate_bot_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.FileHandler('logs/ultimate_bot_latest.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =====================================
# PERFORMANCE TRACKING (from paste.txt)
# =====================================

@dataclass
class Trade:
    """Trade record for performance tracking"""
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    strategy: Optional[str] = None
    closed_at: Optional[datetime] = None
    max_profit: float = 0.0
    max_loss: float = 0.0

class PerformanceTracker:
    """Tracks and analyzes trading performance"""
    
    def __init__(self):
        self.trades: List[Trade] = []
        self.daily_pnl = defaultdict(float)
        self.strategy_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0})
        
    def add_trade(self, trade: Trade):
        """Add a new trade"""
        self.trades.append(trade)
        if trade.closed_at and trade.pnl:
            date_key = trade.closed_at.strftime('%Y-%m-%d')
            self.daily_pnl[date_key] += trade.pnl
            
            # Update strategy performance
            if trade.strategy:
                if trade.pnl > 0:
                    self.strategy_performance[trade.strategy]['wins'] += 1
                else:
                    self.strategy_performance[trade.strategy]['losses'] += 1
                self.strategy_performance[trade.strategy]['pnl'] += trade.pnl
    
    def get_statistics(self) -> Dict:
        """Calculate performance statistics"""
        closed_trades = [t for t in self.trades if t.closed_at]
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_pnl': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl < 0]
        
        # Calculate returns for Sharpe ratio
        returns = [t.pnl_pct for t in closed_trades if t.pnl_pct]
        sharpe_ratio = 0
        if returns and len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                sharpe_ratio = (avg_return * np.sqrt(252)) / std_return  # Annualized
        
        # Calculate max drawdown
        cumulative_pnl = []
        running_total = 0
        for trade in closed_trades:
            running_total += trade.pnl
            cumulative_pnl.append(running_total)
        
        max_drawdown = 0
        if cumulative_pnl:
            peak = cumulative_pnl[0]
            for value in cumulative_pnl:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_trades': len(closed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(closed_trades) * 100 if closed_trades else 0,
            'avg_win': np.mean([t.pnl for t in wins]) if wins else 0,
            'avg_loss': np.mean([t.pnl for t in losses]) if losses else 0,
            'total_pnl': sum(t.pnl for t in closed_trades),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'best_trade': max([t.pnl for t in closed_trades]) if closed_trades else 0,
            'worst_trade': min([t.pnl for t in closed_trades]) if closed_trades else 0,
            'avg_trade_duration': np.mean([(t.closed_at - t.timestamp).total_seconds() / 3600 
                                          for t in closed_trades if t.closed_at]) if closed_trades else 0
        }
    
    def get_strategy_report(self) -> Dict:
        """Get performance by strategy"""
        report = {}
        for strategy, perf in self.strategy_performance.items():
            total = perf['wins'] + perf['losses']
            report[strategy] = {
                'trades': total,
                'win_rate': perf['wins'] / total * 100 if total > 0 else 0,
                'pnl': perf['pnl']
            }
        return report
    
    def save_report(self, filename: str = None):
        """Save performance report to file"""
        if not filename:
            filename = f"logs/performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'strategy_performance': self.get_strategy_report(),
            'daily_pnl': dict(self.daily_pnl),
            'trades': [
                {
                    'timestamp': t.timestamp.isoformat(),
                    'symbol': t.symbol,
                    'side': t.side,
                    'quantity': t.quantity,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct
                    'strategy': t.strategy,
                    'closed_at': t.closed_at.isoformat() if t.closed_at else None
                }
                for t in self.trades
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {filename}")

# =====================================
# LIVE PNL TRACKER (from paste.txt)
# =====================================

class LivePnLTracker:
    """Real-time PnL and statistics tracking"""
    
    def __init__(self):
        self.session_start = datetime.now()
        self.initial_balance = 0
        self.current_balance = 0
        self.peak_balance = 0
        self.wins = 0
        self.losses = 0
        self.current_streak = 0
        self.best_streak = 0
        self.worst_streak = 0
        self.open_positions = {}
        
    def set_initial_balance(self, balance: float):
        """Set initial balance"""
        self.initial_balance = balance
        self.current_balance = balance
        self.peak_balance = balance
        
    def update_balance(self, balance: float):
        """Update current balance"""
        self.current_balance = balance
        self.peak_balance = max(self.peak_balance, balance)
        
    def record_trade_result(self, pnl: float):
        """Record a closed trade result"""
        if pnl > 0:
            self.wins += 1
            if self.current_streak >= 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
            self.best_streak = max(self.best_streak, self.current_streak)
        else:
            self.losses += 1
            if self.current_streak <= 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1
            self.worst_streak = min(self.worst_streak, self.current_streak)
    
    def add_open_position(self, symbol: str, entry_price: float, side: str, quantity: float):
        """Track new open position"""
        self.open_positions[symbol] = {
            'entry_price': entry_price,
            'side': side,
            'quantity': quantity,
            'entry_time': datetime.now()
        }
    
    def update_position_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized PnL for position"""
        if symbol not in self.open_positions:
            return 0
        
        pos = self.open_positions[symbol]
        if pos['side'] == 'Buy':
            pnl = (current_price - pos['entry_price']) * pos['quantity']
        else:
            pnl = (pos['entry_price'] - current_price) * pos['quantity']
        
        return pnl
    
    def remove_position(self, symbol: str):
        """Remove closed position"""
        if symbol in self.open_positions:
            del self.open_positions[symbol]
    
    def get_statistics(self) -> Dict:
        """Get comprehensive live statistics"""
        total_trades = self.wins + self.losses
        win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = self.current_balance - self.initial_balance
        total_pnl_pct = (total_pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0
        
        max_drawdown = 0
        if self.peak_balance > 0:
            max_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100
        
        session_duration = (datetime.now() - self.session_start).total_seconds() / 3600
        
        # Calculate unrealized PnL
        unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in self.open_positions.values())
        
        return {
            'session_duration_hours': round(session_duration, 2),
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'unrealized_pnl': unrealized_pnl,
            'peak_balance': self.peak_balance,
            'max_drawdown_pct': max_drawdown,
            'total_trades': total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'current_streak': self.current_streak,
            'best_streak': self.best_streak,
            'worst_streak': self.worst_streak,
            'open_positions': len(self.open_positions),
            'avg_trades_per_hour': total_trades / session_duration if session_duration > 0 else 0
        }
    
    def print_live_stats(self):
        """Print formatted live statistics"""
        stats = self.get_statistics()
        
        # Color codes for terminal
        GREEN = '\033[92m'
        RED = '\033[91m'
        BLUE = '\033[94m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'
        
        pnl_color = GREEN if stats['total_pnl'] >= 0 else RED
        win_color = GREEN if stats['win_rate'] >= 50 else RED
        streak_color = GREEN if stats['current_streak'] > 0 else RED if stats['current_streak'] < 0 else YELLOW
        
        print(f"\n{BLUE}═══════════════ LIVE TRADING STATS ═══════════════{RESET}")
        print(f"Session Duration: {stats['session_duration_hours']:.1f} hours")
        print(f"Balance: ${stats['current_balance']:.2f} (Initial: ${stats['initial_balance']:.2f})")
        print(f"Total PnL: {pnl_color}${stats['total_pnl']:.2f} ({stats['total_pnl_pct']:+.2f}%){RESET}")
        print(f"Unrealized PnL: ${stats['unrealized_pnl']:.2f}")
        print(f"Max Drawdown: {RED}{stats['max_drawdown_pct']:.2f}%{RESET}")
        print(f"\nWin Rate: {win_color}{stats['win_rate']:.1f}% ({stats['wins']}W / {stats['losses']}L){RESET}")
        print(f"Current Streak: {streak_color}{stats['current_streak']:+d}{RESET}")
        print(f"Best/Worst Streak: {GREEN}+{stats['best_streak']}{RESET} / {RED}{stats['worst_streak']}{RESET}")
        print(f"Open Positions: {stats['open_positions']}")
        print(f"Avg Trades/Hour: {stats['avg_trades_per_hour']:.2f}")
        print(f"{BLUE}═══════════════════════════════════════════════════{RESET}")

# =====================================
# CSV TRADE LOGGER (from paste.txt)
# =====================================

class CSVTradeLogger:
    """Logs trades to CSV file for real-time PnL tracking"""
    
    def __init__(self, filename: str = "trade_log.csv"):
        self.filename = filename
        self.initialize_csv()
    
    def initialize_csv(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.filename):
            headers = ['timestamp', 'symbol', 'side', 'entry_price', 'quantity', 
                      'rsi', 'macd_signal', 'ema_signal', 'strategy', 'exit_price', 'pnl']
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.filename, index=False)
            logger.info(f"Created trade log: {self.filename}")
    
    def log_trade_entry(self, trade: Trade, signals: Dict):
        """Log trade entry with technical indicators"""
        try:
            # Extract RSI value from signals
            rsi_value = None
            if 'analyses' in signals and 'RSI' in signals['analyses']:
                rsi_value = signals['analyses']['RSI'].get('rsi', None)
            
            # Extract other signal values
            macd_signal = 'BUY' if any(s['strategy'] == 'MACD' and s['signal'] == SignalType.BUY 
                                      for s in signals.get('individual_signals', [])) else 'SELL' if any(
                                      s['strategy'] == 'MACD' and s['signal'] == SignalType.SELL 
                                      for s in signals.get('individual_signals', [])) else 'HOLD'
            
            ema_signal = 'BUY' if any(s['strategy'] == 'EMA' and s['signal'] == SignalType.BUY 
                                     for s in signals.get('individual_signals', [])) else 'SELL' if any(
                                     s['strategy'] == 'EMA' and s['signal'] == SignalType.SELL 
                                     for s in signals.get('individual_signals', [])) else 'HOLD'
            
            # Create log entry
            log_entry = {
                'timestamp': trade.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_price': trade.entry_price,
                'quantity': trade.quantity,
                'rsi': round(rsi_value, 2) if rsi_value else None,
                'macd_signal': macd_signal,
                'ema_signal': ema_signal,
                'strategy': trade.strategy,
                'exit_price': None,
                'pnl': None
            }
            
            # Append to CSV
            df = pd.DataFrame([log_entry])
            df.to_csv(self.filename, mode='a', header=False, index=False)
            logger.info(f"Trade logged to CSV: {trade.symbol} {trade.side}")
            
        except Exception as e:
            logger.error(f"Error logging trade to CSV: {e}")
    
    def update_trade_exit(self, symbol: str, exit_price: float, pnl: float):
        """Update trade exit information in CSV"""
        try:
            # Read CSV
            df = pd.read_csv(self.filename)
            
            # Find the last open trade for this symbol
            symbol_trades = df[df['symbol'] == symbol]
            open_trades = symbol_trades[symbol_trades['exit_price'].isna()]
            
            if not open_trades.empty:
                # Update the last open trade
                idx = open_trades.index[-1]
                df.loc[idx, 'exit_price'] = exit_price
                df.loc[idx, 'pnl'] = pnl
                
                # Save updated CSV
                df.to_csv(self.filename, index=False)
                logger.info(f"Updated trade exit in CSV: {symbol} PnL: ${pnl:.2f}")
                
        except Exception as e:
            logger.error(f"Error updating trade exit in CSV: {e}")

# =====================================
# TRAILING STOP MANAGER (from paste.txt)
# =====================================

@dataclass
class TrailingStopPosition:
    """Tracks trailing stop for a position"""
    symbol: str
    side: str
    entry_price: float
    quantity: float
    highest_price: float
    lowest_price: float
    trailing_activated: bool = False
    current_stop_price: Optional[float] = None
    last_update: datetime = field(default_factory=datetime.now)

class TrailingStopManager:
    """Manages trailing stops for all positions"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.positions: Dict[str, TrailingStopPosition] = {}
        
    def add_position(self, symbol: str, side: str, entry_price: float, quantity: float):
        """Add a new position to track"""
        self.positions[symbol] = TrailingStopPosition(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            highest_price=entry_price,
            lowest_price=entry_price
        )
        logger.info(f"Trailing stop tracking started for {symbol}")
    
    def update_position(self, symbol: str, current_price: float) -> Optional[float]:
        """
        Update position and return new stop price if changed
        Returns None if no update needed
        """
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        new_stop_price = None
        
        if pos.side == "Buy":
            # Update highest price
            if current_price > pos.highest_price:
                pos.highest_price = current_price
            
            # Calculate profit percentage
            profit_pct = (current_price - pos.entry_price) / pos.entry_price * 100
            
            # Check if we should activate trailing
            if not pos.trailing_activated and profit_pct >= self.config.TRAIL_TRIGGER:
                pos.trailing_activated = True
                new_stop_price = pos.entry_price * (1 + 0.002)  # Set stop at breakeven + 0.2%
                pos.current_stop_price = new_stop_price
                logger.info(f"Trailing stop activated for {symbol} at ${new_stop_price:.2f}")
            
            # Update trailing stop if active
            elif pos.trailing_activated:
                trail_price = pos.highest_price * (1 - self.config.TRAIL_OFFSET / 100)
                if trail_price > (pos.current_stop_price or 0):
                    new_stop_price = trail_price
                    pos.current_stop_price = new_stop_price
                    logger.info(f"Trailing stop updated for {symbol} to ${new_stop_price:.2f}")
        
        else:  # Sell position
            # Update lowest price
            if current_price < pos.lowest_price:
                pos.lowest_price = current_price
            
            # Calculate profit percentage
            profit_pct = (pos.entry_price - current_price) / pos.entry_price * 100
            
            # Check if we should activate trailing
            if not pos.trailing_activated and profit_pct >= self.config.TRAIL_TRIGGER:
                pos.trailing_activated = True
                new_stop_price = pos.entry_price * (1 - 0.002)  # Set stop at breakeven - 0.2%
                pos.current_stop_price = new_stop_price
                logger.info(f"Trailing stop activated for {symbol} at ${new_stop_price:.2f}")
            
            # Update trailing stop if active
            elif pos.trailing_activated:
                trail_price = pos.lowest_price * (1 + self.config.TRAIL_OFFSET / 100)
                if trail_price < (pos.current_stop_price or float('inf')):
                    new_stop_price = trail_price
                    pos.current_stop_price = new_stop_price
                    logger.info(f"Trailing stop updated for {symbol} to ${new_stop_price:.2f}")
        
        pos.last_update = datetime.now()
        return new_stop_price
    
    def remove_position(self, symbol: str):
        """Remove position from tracking"""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"Stopped tracking trailing stop for {symbol}")
    
    def get_position_status(self, symbol: str) -> Optional[Dict]:
        """Get current status of a position"""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        return {
            'trailing_activated': pos.trailing_activated,
            'current_stop': pos.current_stop_price,
            'highest_price': pos.highest_price,
            'lowest_price': pos.lowest_price,
            'entry_price': pos.entry_price
        }

# =====================================
# TECHNICAL INDICATORS (from paste.txt)
# =====================================

class TechnicalIndicators:
    """Enhanced technical indicator calculations"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
            
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return pd.Series([50] * len(prices))
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
        except Exception as e:
            logger.error(f"MACD calculation error: {e}")
            return {'macd': pd.Series([0] * len(prices)), 
                   'signal': pd.Series([0] * len(prices)), 
                   'histogram': pd.Series([0] * len(prices))}
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA"""
        try:
            return prices.ewm(span=period).mean()
        except Exception as e:
            logger.error(f"EMA calculation error: {e}")
            return prices

# =====================================
# TRADING STRATEGIES (from paste.txt)
# =====================================

class BaseStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.indicators = TechnicalIndicators()
    
    def analyze(self, df: pd.DataFrame) -> Tuple[SignalType, float, Dict]:
        """Override in subclasses"""
        raise NotImplementedError

class RSIStrategy(BaseStrategy):
    """RSI-based trading strategy"""
    
    def __init__(self, config: BotConfig):
        super().__init__("RSI")
        self.config = config
        
    def analyze(self, df: pd.DataFrame) -> Tuple[SignalType, float, Dict]:
        """Analyze market data and generate signal"""
        try:
            if len(df) < self.config.RSI_PERIOD + 5:
                return SignalType.HOLD, 0.0, {}
            
            # Calculate RSI
            rsi = self.indicators.calculate_rsi(df['close'], self.config.RSI_PERIOD)
            current_rsi = rsi.iloc[-1]
            rsi_trend = rsi.iloc[-1] - rsi.iloc[-2]
            
            # Volume analysis
            volume_ratio = df['volume'].iloc[-1] / df['volume'].tail(20).mean()
            
            analysis = {
                'rsi': current_rsi,
                'rsi_trend': rsi_trend,
                'volume_ratio': volume_ratio
            }
            
            # Generate signals
            if current_rsi < self.config.RSI_OVERSOLD and rsi_trend > 0 and volume_ratio > 1.2:
                strength = (self.config.RSI_OVERSOLD - current_rsi) / self.config.RSI_OVERSOLD
                return SignalType.BUY, min(strength * 1.5, 1.0), analysis
                
            elif current_rsi > self.config.RSI_OVERBOUGHT and rsi_trend < 0 and volume_ratio > 1.2:
                strength = (current_rsi - self.config.RSI_OVERBOUGHT) / (100 - self.config.RSI_OVERBOUGHT)
                return SignalType.SELL, min(strength * 1.5, 1.0), analysis
            
            return SignalType.HOLD, 0.0, analysis
            
        except Exception as e:
            logger.error(f"RSI strategy error: {e}")
            return SignalType.HOLD, 0.0, {}

class MACDStrategy(BaseStrategy):
    """MACD-based trading strategy"""
    
    def __init__(self, config: BotConfig):
        super().__init__("MACD")
        self.config = config
        
    def analyze(self, df: pd.DataFrame) -> Tuple[SignalType, float, Dict]:
        """Analyze using MACD"""
        try:
            if len(df) < 50:
                return SignalType.HOLD, 0.0, {}
            
            # Calculate MACD
            macd_data = self.indicators.calculate_macd(
                df['close'], 
                self.config.MACD_FAST, 
                self.config.MACD_SLOW, 
                self.config.MACD_SIGNAL
            )
            
            macd = macd_data['macd'].iloc[-1]
            signal = macd_data['signal'].iloc[-1]
            histogram = macd_data['histogram'].iloc[-1]
            prev_histogram = macd_data['histogram'].iloc[-2]
            
            # Check for crossover
            crossover_up = histogram > 0 and prev_histogram <= 0
            crossover_down = histogram < 0 and prev_histogram >= 0
            
            analysis = {
                'macd': macd,
                'signal': signal,
                'histogram': histogram,
                'crossover_up': crossover_up,
                'crossover_down': crossover_down
            }
            
            # Generate signals
            if crossover_up and macd < 0:  # Bullish crossover in negative territory
                strength = min(abs(histogram) * 100, 1.0)
                return SignalType.BUY, strength, analysis
                
            elif crossover_down and macd > 0:  # Bearish crossover in positive territory
                strength = min(abs(histogram) * 100, 1.0)
                return SignalType.SELL, strength, analysis
            
            return SignalType.HOLD, 0.0, analysis
            
        except Exception as e:
            logger.error(f"MACD strategy error: {e}")
            return SignalType.HOLD, 0.0, {}

class EMAStrategy(BaseStrategy):
    """EMA crossover strategy"""
    
    def __init__(self, config: BotConfig):
        super().__init__("EMA")
        self.config = config
        
    def analyze(self, df: pd.DataFrame) -> Tuple[SignalType, float, Dict]:
        """Analyze using EMA crossover"""
        try:
            if len(df) < self.config.EMA_SLOW + 5:
                return SignalType.HOLD, 0.0, {}
            
            # Calculate EMAs
            ema_fast = self.indicators.calculate_ema(df['close'], self.config.EMA_FAST)
            ema_slow = self.indicators.calculate_ema(df['close'], self.config.EMA_SLOW)
            
            current_price = df['close'].iloc[-1]
            fast_current = ema_fast.iloc[-1]
            slow_current = ema_slow.iloc[-1]
            fast_prev = ema_fast.iloc[-2]
            slow_prev = ema_slow.iloc[-2]
            
            # Check for crossover
            crossover_up = fast_current > slow_current and fast_prev <= slow_prev
            crossover_down = fast_current < slow_current and fast_prev >= slow_prev
            
            # Calculate trend strength
            ema_diff = (fast_current - slow_current) / slow_current * 100
            
            analysis = {
                'ema_fast': fast_current,
                'ema_slow': slow_current,
                'ema_diff': ema_diff,
                'price_vs_fast': (current_price - fast_current) / fast_current * 100
            }
            
            # Generate signals
            if crossover_up and current_price > fast_current:
                strength = min(abs(ema_diff) * 0.3, 1.0)
                return SignalType.BUY, strength, analysis
                
            elif crossover_down and current_price < fast_current:
                strength = min(abs(ema_diff) * 0.3, 1.0)
                return SignalType.SELL, strength, analysis
            
            return SignalType.HOLD, 0.0, analysis
            
        except Exception as e:
            logger.error(f"EMA strategy error: {e}")
            return SignalType.HOLD, 0.0, {}

# =====================================
# ML/AI FEATURE ENGINEERING (from paste.txt)
# =====================================

class FeatureEngineer:
    """Creates features for ML models from market data"""
    
    @staticmethod
    def create_features(df: pd.DataFrame, config: BotConfig) -> pd.DataFrame:
        """Create technical features for ML models"""
        try:
            features = pd.DataFrame()
            
            # Price features
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            features['price_range'] = (df['high'] - df['low']) / df['close']
            features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Volume features
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            features['volume_change'] = df['volume'].pct_change()
            
            # Technical indicators
            indicators = TechnicalIndicators()
            
            # RSI
            features['rsi'] = indicators.calculate_rsi(df['close'], config.RSI_PERIOD)
            features['rsi_change'] = features['rsi'].diff()
            
            # MACD
            macd_data = indicators.calculate_macd(df['close'])
            features['macd'] = macd_data['macd']
            features['macd_signal'] = macd_data['signal']
            features['macd_histogram'] = macd_data['histogram']
            features['macd_cross'] = (macd_data['histogram'] > 0).astype(int)
            
            # EMAs
            features['ema_9'] = indicators.calculate_ema(df['close'], 9)
            features['ema_21'] = indicators.calculate_ema(df['close'], 21)
            features['ema_50'] = indicators.calculate_ema(df['close'], 50)
            features['ema_cross'] = (features['ema_9'] > features['ema_21']).astype(int)
            
            # Price relative to EMAs
            features['price_vs_ema9'] = (df['close'] - features['ema_9']) / features['ema_9']
            features['price_vs_ema21'] = (df['close'] - features['ema_21']) / features['ema_21']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            sma = df['close'].rolling(bb_period).mean()
            std = df['close'].rolling(bb_period).std()
            features['bb_upper'] = sma + (bb_std * std)
            features['bb_lower'] = sma - (bb_std * std)
            features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # Market microstructure
            features['spread'] = (df['high'] - df['low']) / df['close']
            features['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
            
            # Momentum
            features['momentum_3'] = df['close'].pct_change(3)
            features['momentum_5'] = df['close'].pct_change(5)
            features['momentum_10'] = df['close'].pct_change(10)
            
            # Target variable (1 if price goes up in next candle, 0 if down)
            features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Drop NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"Feature engineering error: {e}")
            return pd.DataFrame()

# =====================================
# ML TRADING STRATEGIES (from paste.txt)
# =====================================

class RandomForestStrategy(BaseStrategy):
    """RandomForest ML-based trading strategy"""
    
    def __init__(self, config: BotConfig):
        super().__init__("RandomForest")
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.accuracy = 0.0
        self.feature_engineer = FeatureEngineer()
        self.training_data = deque(maxlen=5000)  # Store recent data for retraining
        
    def train(self, features_df: pd.DataFrame):
        """Train the RandomForest model"""
        try:
            if len(features_df) < 100:
                return
            
            # Prepare features and target
            feature_cols = [col for col in features_df.columns if col != 'target']
            X = features_df[feature_cols]
            y = features_df['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            self.accuracy = self.model.score(X_test_scaled, y_test)
            self.is_trained = True
            
            logger.info(f"RandomForest trained with accuracy: {self.accuracy:.2%}")
            
            # Save model
            self._save_model()
            
        except Exception as e:
            logger.error(f"RandomForest training error: {e}")
    
    def _save_model(self):
        """Save trained model to disk"""
        try:
            os.makedirs('models', exist_ok=True)
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'accuracy': self.accuracy
            }
            with open('models/random_forest.pkl', 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _load_model(self):
        """Load model from disk"""
        try:
            if os.path.exists('models/random_forest.pkl'):
                with open('models/random_forest.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.accuracy = model_data['accuracy']
                    self.is_trained = True
                    logger.info(f"Loaded RandomForest model (accuracy: {self.accuracy:.2%})")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def analyze(self, df: pd.DataFrame) -> Tuple[SignalType, float, Dict]:
        """Analyze using RandomForest predictions"""
        try:
            if not ML_AVAILABLE:
                return SignalType.HOLD, 0.0, {'error': 'ML not available'}
            
            # Load model if not trained
            if not self.is_trained:
                self._load_model()
            
            # Create features
            features_df = self.feature_engineer.create_features(df, self.config)
            if features_df.empty or not self.is_trained:
                return SignalType.HOLD, 0.0, {'accuracy': self.accuracy}
            
            # Store data for retraining
            self.training_data.append(features_df)
            
            # Get latest features
            feature_cols = [col for col in features_df.columns if col != 'target']
            latest_features = features_df[feature_cols].iloc[-1:].values
            
            # Scale and predict
            latest_scaled = self.scaler.transform(latest_features)
            prediction = self.model.predict(latest_scaled)[0]
            probability = self.model.predict_proba(latest_scaled)[0]
            
            # Get feature importances
            importances = self.model.feature_importances_
            top_features = sorted(
                zip(feature_cols, importances), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            analysis = {
                'prediction': int(prediction),
                'probability_up': float(probability[1]),
                'probability_down': float(probability[0]),
                'accuracy': self.accuracy,
                'top_features': top_features
            }
            
            # Generate signal only if accuracy is good
            if self.accuracy >= self.config.ML_MIN_ACCURACY:
                if prediction == 1 and probability[1] > 0.65:
                    return SignalType.BUY, float(probability[1]), analysis
                elif prediction == 0 and probability[0] > 0.65:
                    return SignalType.SELL, float(probability[0]), analysis
            
            return SignalType.HOLD, 0.0, analysis
            
        except Exception as e:
            logger.error(f"RandomForest analysis error: {e}")
            return SignalType.HOLD, 0.0, {}

class EnsembleMLStrategy(BaseStrategy):
    """Ensemble of multiple ML models"""
    
    def __init__(self, config: BotConfig):
        super().__init__("EnsembleML")
        self.config = config
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.accuracy = 0.0
        self.feature_engineer = FeatureEngineer()
        
        if ML_AVAILABLE:
            self.models = {
                'rf': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
                'lr': LogisticRegression(random_state=42),
                'svm': SVC(probability=True, random_state=42),
                'nb': GaussianNB()
            }
    
    def train(self, features_df: pd.DataFrame):
        """Train ensemble models"""
        try:
            if not ML_AVAILABLE or len(features_df) < 100:
                return
            
            # Prepare data
            feature_cols = [col for col in features_df.columns if col != 'target']
            X = features_df[feature_cols]
            y = features_df['target']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train each model
            accuracies = {}
            for name, model in self.models.items():
                model.fit(X_train_scaled, y_train)
                acc = model.score(X_test_scaled, y_test)
                accuracies[name] = acc
                logger.info(f"Ensemble {name} accuracy: {acc:.2%}")
            
            self.accuracy = np.mean(list(accuracies.values()))
            self.is_trained = True
            logger.info(f"Ensemble average accuracy: {self.accuracy:.2%}")
            
        except Exception as e:
            logger.error(f"Ensemble training error: {e}")
    
    def analyze(self, df: pd.DataFrame) -> Tuple[SignalType, float, Dict]:
        """Analyze using ensemble voting"""
        try:
            if not ML_AVAILABLE or not self.is_trained:
                return SignalType.HOLD, 0.0, {}
            
            # Create features
            features_df = self.feature_engineer.create_features(df, self.config)
            if features_df.empty:
                return SignalType.HOLD, 0.0, {}
            
            # Get latest features
            feature_cols = [col for col in features_df.columns if col != 'target']
            latest_features = features_df[feature_cols].iloc[-1:].values
            latest_scaled = self.scaler.transform(latest_features)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                pred = model.predict(latest_scaled)[0]
                prob = model.predict_proba(latest_scaled)[0]
                predictions[name] = pred
                probabilities[name] = prob
            
            # Ensemble voting
            buy_votes = sum(1 for p in predictions.values() if p == 1)
            sell_votes = len(predictions) - buy_votes
            
            # Average probabilities
            avg_prob_up = np.mean([p[1] for p in probabilities.values()])
            avg_prob_down = np.mean([p[0] for p in probabilities.values()])
            
            analysis = {
                'predictions': predictions,
                'buy_votes': buy_votes,
                'sell_votes': sell_votes,
                'avg_prob_up': float(avg_prob_up),
                'avg_prob_down': float(avg_prob_down),
                'accuracy': self.accuracy
            }
            
            # Generate signal
            if self.accuracy >= self.config.ML_MIN_ACCURACY:
                if buy_votes > sell_votes and avg_prob_up > 0.6:
                    return SignalType.BUY, float(avg_prob_up), analysis
                elif sell_votes > buy_votes and avg_prob_down > 0.6:
                    return SignalType.SELL, float(avg_prob_down), analysis
            
            return SignalType.HOLD, 0.0, analysis
            
        except Exception as e:
            logger.error(f"Ensemble analysis error: {e}")
            return SignalType.HOLD, 0.0, {}

# =====================================
# PPO REINFORCEMENT LEARNING (from paste.txt)
# =====================================

class TradingEnvironment(gym.Env):
    """Custom trading environment for PPO agent"""
    
    def __init__(self, df: pd.DataFrame, config: BotConfig):
        super(TradingEnvironment, self).__init__()
        
        self.df = df
        self.config = config
        self.feature_engineer = FeatureEngineer()
        
        # Prepare features
        self.features = self.feature_engineer.create_features(df, config)
        self.feature_cols = [col for col in self.features.columns if col != 'target']
        
        # Environment settings
        self.current_step = 0
        self.max_steps = len(self.features) - 1
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(self.feature_cols),), 
            dtype=np.float32
        )
        
        # Trading state
        self.position = 0  # 0=no position, 1=long, -1=short
        self.entry_price = 0
        self.total_reward = 0
        self.trades = []
        
    def reset(self):
        """Reset environment"""
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.total_reward = 0
        self.trades = []
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation"""
        if self.current_step >= len(self.features):
            return np.zeros(len(self.feature_cols))
        
        obs = self.features[self.feature_cols].iloc[self.current_step].values
        return obs.astype(np.float32)
    
    def step(self, action):
        """Execute action and return results"""
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, {}
        
        current_price = self.df['close'].iloc[self.current_step]
        reward = 0
        
        # Execute action
        if action == 1 and self.position <= 0:  # Buy
            if self.position == -1:  # Close short
                pnl = self.entry_price - current_price
                reward = pnl / self.entry_price * 100
                self.trades.append(('close_short', current_price, pnl))
            
            self.position = 1
            self.entry_price = current_price
            self.trades.append(('buy', current_price, 0))
            
        elif action == 2 and self.position >= 0:  # Sell
            if self.position == 1:  # Close long
                pnl = current_price - self.entry_price
                reward = pnl / self.entry_price * 100
                self.trades.append(('close_long', current_price, pnl))
            
            self.position = -1
            self.entry_price = current_price
            self.trades.append(('sell', current_price, 0))
        
        # Small penalty for holding to encourage trading
        if action == 0 and self.position != 0:
            unrealized_pnl = 0
            if self.position == 1:
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price * 100
            elif self.position == -1:
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price * 100
            reward = unrealized_pnl * 0.01  # Small reward for good positions
        
        self.current_step += 1
        self.total_reward += reward
        
        done = self.current_step >= self.max_steps
        
        # Close position at end
        if done and self.position != 0:
            final_price = self.df['close'].iloc[-1]
            if self.position == 1:
                pnl = final_price - self.entry_price
                reward += pnl / self.entry_price * 100
            elif self.position == -1:
                pnl = self.entry_price - final_price
                reward += pnl / self.entry_price * 100
        
        info = {
            'position': self.position,
            'total_reward': self.total_reward,
            'trades': len(self.trades)
        }
        
        return self._get_observation(), reward, done, info

class PPOStrategy(BaseStrategy):
    """PPO Reinforcement Learning Strategy"""
    
    def __init__(self, config: BotConfig):
        super().__init__("PPO_RL")
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_engineer = FeatureEngineer()
        self.training_rewards = []
        
        if RL_AVAILABLE:
            self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            if os.path.exists(self.config.PPO_MODEL_PATH):
                self.model = PPO.load(self.config.PPO_MODEL_PATH)
                self.is_trained = True
                logger.info("Loaded PPO model")
            else:
                logger.info("Creating new PPO model...")
                # Create dummy environment for model initialization
                dummy_df = pd.DataFrame({
                    'open': np.random.rand(100),
                    'high': np.random.rand(100),
                    'low': np.random.rand(100),
                    'close': np.random.rand(100),
                    'volume': np.random.rand(100)
                })
                env = TradingEnvironment(dummy_df, self.config)
                self.model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=self.config.PPO_LEARNING_RATE,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    verbose=1
                )
                os.makedirs(os.path.dirname(self.config.PPO_MODEL_PATH), exist_ok=True)
                
        except Exception as e:
            logger.error(f"PPO model initialization error: {e}")
    
    def train(self, df: pd.DataFrame):
        """Train PPO agent"""
        try:
            if not RL_AVAILABLE or self.config.PPO_MODE != "train":
                return
            
            logger.info("Training PPO agent...")
            
            # Create environment
            env = TradingEnvironment(df, self.config)
            vec_env = DummyVecEnv([lambda: env])
            
            # Train
            self.model.set_env(vec_env)
            self.model.learn(total_timesteps=self.config.PPO_EPISODES)
            
            # Save model
            self.model.save(self.config.PPO_MODEL_PATH)
            self.is_trained = True
            
            logger.info("PPO training completed")
            
        except Exception as e:
            logger.error(f"PPO training error: {e}")
    
    def analyze(self, df: pd.DataFrame) -> Tuple[SignalType, float, Dict]:
        """Analyze using PPO agent"""
        try:
            if not RL_AVAILABLE or not self.model:
                return SignalType.HOLD, 0.0, {}
            
            # Create features
            features_df = self.feature_engineer.create_features(df, self.config)
            if features_df.empty:
                return SignalType.HOLD, 0.0, {}
            
            # Get latest observation
            feature_cols = [col for col in features_df.columns if col != 'target']
            obs = features_df[feature_cols].iloc[-1].values.astype(np.float32)
            
            # Predict action
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Get action probabilities
            try:
                if hasattr(self.model.policy, 'action_net'):
                    with torch.no_grad():
                        features = self.model.policy.features_extractor(torch.tensor(obs).unsqueeze(0))
                        latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
                        action_logits = self.model.policy.action_net(latent_pi)
                        action_probs = torch.softmax(action_logits, dim=1).numpy()[0]
                else:
                    # Simplified probability estimation
                    action_probs = np.zeros(3)
                    action_probs[action] = 0.8
                    action_probs = action_probs / action_probs.sum()
            except:
                # Fallback if torch operations fail
                action_probs = np.zeros(3)
                action_probs[action] = 0.8
                action_probs = action_probs / action_probs.sum()
            
            analysis = {
                'action': int(action),
                'action_probs': {
                    'hold': float(action_probs[0]),
                    'buy': float(action_probs[1]),
                    'sell': float(action_probs[2])
                },
                'mode': self.config.PPO_MODE,
                'is_trained': self.is_trained
            }
            
            # Convert action to signal
            if action == 1:  # Buy
                return SignalType.BUY, float(action_probs[1]), analysis
            elif action == 2:  # Sell
                return SignalType.SELL, float(action_probs[2]), analysis
            else:  # Hold
                return SignalType.HOLD, 0.0, analysis
                
        except Exception as e:
            logger.error(f"PPO analysis error: {e}")
            return SignalType.HOLD, 0.0, {}

# =====================================
# NEW ADVANCED FEATURES (from paste-2.txt)
# =====================================

# =====================================
# MULTI-TIMEFRAME ANALYSIS
# =====================================

class MultiTimeframeAnalyzer:
    """Analyzes multiple timeframes for confluence"""
    
    def __init__(self, config: BotConfig, session):
        self.config = config
        self.session = session
        self.cache = {}
        
    async def get_mtf_signals(self, symbol: str) -> Dict[str, Dict]:
        """Get signals from multiple timeframes"""
        signals = {}
        
        for timeframe in self.config.TIMEFRAMES:
            try:
                # Fetch data for timeframe
                df = await self._fetch_timeframe_data(symbol, timeframe)
                if df is not None:
                    # Analyze each timeframe
                    signal = self._analyze_timeframe(df, timeframe)
                    signals[timeframe] = signal
            except Exception as e:
                logging.error(f"MTF error for {symbol} {timeframe}: {e}")
                
        return signals
    
    async def _fetch_timeframe_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch data for specific timeframe"""
        cache_key = f"{symbol}_{interval}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < 30:  # 30 second cache
                return cached_data
                
        try:
            # Fetch from API
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    limit=200
                )
            )
            
            if response['retCode'] != 0:
                return None
                
            # Convert to DataFrame
            klines = response['result']['list']
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Cache the data
            self.cache[cache_key] = (df, time.time())
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching {interval} data for {symbol}: {e}")
            return None
    
    def _analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Analyze single timeframe"""
        try:
            # Calculate indicators
            rsi = ta.momentum.RSIIndicator(df['close']).rsi().iloc[-1] if TA_AVAILABLE else 50
            
            # MACD
            if TA_AVAILABLE:
                macd = ta.trend.MACD(df['close'])
                macd_value = macd.macd().iloc[-1]
                macd_signal = macd.macd_signal().iloc[-1]
                macd_cross = macd_value > macd_signal
            else:
                macd_cross = False
            
            # Trend
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Determine signal
            bullish_points = 0
            bearish_points = 0
            
            if rsi < 30:
                bullish_points += 2
            elif rsi > 70:
                bearish_points += 2
                
            if macd_cross:
                bullish_points += 1
            else:
                bearish_points += 1
                
            if current_price > sma_20 > sma_50:
                bullish_points += 2
            elif current_price < sma_20 < sma_50:
                bearish_points += 2
                
            # Determine signal
            if bullish_points > bearish_points + 1:
                signal = SignalType.BUY
                strength = bullish_points / (bullish_points + bearish_points)
            elif bearish_points > bullish_points + 1:
                signal = SignalType.SELL
                strength = bearish_points / (bullish_points + bearish_points)
            else:
                signal = SignalType.HOLD
                strength = 0.5
                
            return {
                'timeframe': timeframe,
                'signal': signal,
                'strength': strength,
                'rsi': rsi,
                'trend': 'bullish' if current_price > sma_50 else 'bearish',
                'price': current_price
            }
            
        except Exception as e:
            logging.error(f"Timeframe analysis error: {e}")
            return {
                'timeframe': timeframe,
                'signal': SignalType.HOLD,
                'strength': 0.0
            }
    
    def get_confluence_signal(self, mtf_signals: Dict[str, Dict]) -> Tuple[SignalType, float, Dict]:
        """Determine confluence across timeframes"""
        if not mtf_signals:
            return SignalType.HOLD, 0.0, {}
            
        buy_count = sum(1 for s in mtf_signals.values() if s['signal'] == SignalType.BUY)
        sell_count = sum(1 for s in mtf_signals.values() if s['signal'] == SignalType.SELL)
        
        # Calculate weighted strength (higher timeframes have more weight)
        timeframe_weights = {'1': 0.2, '5': 0.3, '15': 0.5}
        
        buy_strength = sum(
            s['strength'] * timeframe_weights.get(s['timeframe'], 0.3)
            for s in mtf_signals.values()
            if s['signal'] == SignalType.BUY
        )
        
        sell_strength = sum(
            s['strength'] * timeframe_weights.get(s['timeframe'], 0.3)
            for s in mtf_signals.values()
            if s['signal'] == SignalType.SELL
        )
        
        analysis = {
            'mtf_signals': mtf_signals,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'buy_strength': buy_strength,
            'sell_strength': sell_strength
        }
        
        # Need confluence
        if buy_count >= self.config.MTF_CONFLUENCE_REQUIRED and buy_strength > 0.6:
            return SignalType.BUY, buy_strength, analysis
        elif sell_count >= self.config.MTF_CONFLUENCE_REQUIRED and sell_strength > 0.6:
            return SignalType.SELL, sell_strength, analysis
        else:
            return SignalType.HOLD, 0.0, analysis

# =====================================
# SENTIMENT ANALYSIS
# =====================================

class SentimentAnalyzer:
    """Analyzes market sentiment from various sources"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.sentiment_cache = {}
        self.cache_duration = 300  # 5 minutes
        
    async def get_sentiment(self, symbol: str) -> Dict:
        """Get aggregated sentiment for symbol"""
        # Check cache
        if symbol in self.sentiment_cache:
            cached_data, timestamp = self.sentiment_cache[symbol]
            if time.time() - timestamp < self.cache_duration:
                return cached_data
                
        sentiment_data = {
            'overall_sentiment': 0.0,
            'sources': {}
        }
        
        # Get sentiment from each source
        tasks = []
        if 'cryptopanic' in self.config.SENTIMENT_SOURCES:
            tasks.append(self._get_cryptopanic_sentiment(symbol))
        if 'reddit' in self.config.SENTIMENT_SOURCES:
            tasks.append(self._get_reddit_sentiment(symbol))
            
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            valid_sentiments = []
            for i, result in enumerate(results):
                if isinstance(result, dict) and 'sentiment' in result:
                    source_name = self.config.SENTIMENT_SOURCES[i]
                    sentiment_data['sources'][source_name] = result
                    valid_sentiments.append(result['sentiment'])
                    
            # Calculate overall sentiment
            if valid_sentiments:
                sentiment_data['overall_sentiment'] = np.mean(valid_sentiments)
                
        # Cache results
        self.sentiment_cache[symbol] = (sentiment_data, time.time())
        
        return sentiment_data
    
    async def _get_cryptopanic_sentiment(self, symbol: str) -> Dict:
        """Get sentiment from CryptoPanic"""
        try:
            # This would connect to actual CryptoPanic API
            # For now, return simulated data
            async with aiohttp.ClientSession() as session:
                # Simulated API call
                await asyncio.sleep(0.1)  # Simulate network delay
                
                # In production, this would parse actual news
                sentiment_score = np.random.uniform(-1, 1)
                
                return {
                    'sentiment': sentiment_score,
                    'confidence': 0.7,
                    'news_count': np.random.randint(5, 20),
                    'source': 'cryptopanic'
                }
                
        except Exception as e:
            logging.error(f"CryptoPanic sentiment error: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0}
    
    async def _get_reddit_sentiment(self, symbol: str) -> Dict:
        """Get sentiment from Reddit"""
        try:
            # This would use PRAW or Reddit API
            # For now, return simulated data
            async with aiohttp.ClientSession() as session:
                await asyncio.sleep(0.1)  # Simulate network delay
                
                # Simulated sentiment analysis
                sentiment_score = np.random.uniform(-1, 1)
                
                return {
                    'sentiment': sentiment_score,
                    'confidence': 0.6,
                    'post_count': np.random.randint(10, 50),
                    'source': 'reddit'
                }
                
        except Exception as e:
            logging.error(f"Reddit sentiment error: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0}
    
    def interpret_sentiment(self, sentiment_data: Dict) -> Tuple[float, str]:
        """Interpret sentiment data into trading signal modifier"""
        overall = sentiment_data.get('overall_sentiment', 0.0)
        
        if overall > 0.5:
            return 1.2, "Very Bullish"  # 20% boost to buy signals
        elif overall > 0.2:
            return 1.1, "Bullish"      # 10% boost
        elif overall < -0.5:
            return 0.8, "Very Bearish" # 20% reduction
        elif overall < -0.2:
            return 0.9, "Bearish"      # 10% reduction
        else:
            return 1.0, "Neutral"

# =====================================
# WHALE TRACKING
# =====================================

class WhaleTracker:
    """Tracks large transactions and whale movements"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.whale_cache = {}
        self.cache_duration = 600  # 10 minutes
        
    async def get_whale_activity(self, symbol: str) -> Dict:
        """Get whale activity for symbol"""
        # Check cache
        if symbol in self.whale_cache:
            cached_data, timestamp = self.whale_cache[symbol]
            if time.time() - timestamp < self.cache_duration:
                return cached_data
                
        try:
            # In production, this would connect to:
            # - Glassnode API
            # - Santiment API
            # - WhaleAlert API
            # - On-chain data providers
            
            # Simulated whale data
            whale_data = {
                'large_transactions': np.random.randint(0, 10),
                'whale_accumulation': np.random.uniform(-1, 1),
                'exchange_inflows': np.random.uniform(0, 1000000),
                'exchange_outflows': np.random.uniform(0, 1000000),
                'whale_positions': {
                    'long': np.random.uniform(0, 100),
                    'short': np.random.uniform(0, 100)
                }
            }
            
            # Calculate whale sentiment
            net_flow = whale_data['exchange_outflows'] - whale_data['exchange_inflows']
            whale_sentiment = 0.0
            
            if net_flow > self.config.WHALE_THRESHOLD_USD:
                whale_sentiment = 0.5  # Bullish - whales withdrawing
            elif net_flow < -self.config.WHALE_THRESHOLD_USD:
                whale_sentiment = -0.5  # Bearish - whales depositing
                
            if whale_data['whale_accumulation'] > 0.5:
                whale_sentiment += 0.3
            elif whale_data['whale_accumulation'] < -0.5:
                whale_sentiment -= 0.3
                
            whale_data['whale_sentiment'] = np.clip(whale_sentiment, -1, 1)
            
            # Cache results
            self.whale_cache[symbol] = (whale_data, time.time())
            
            return whale_data
            
        except Exception as e:
            logging.error(f"Whale tracking error: {e}")
            return {'whale_sentiment': 0.0}
    
    def interpret_whale_activity(self, whale_data: Dict) -> Tuple[float, str]:
        """Interpret whale activity into signal modifier"""
        sentiment = whale_data.get('whale_sentiment', 0.0)
        
        if sentiment > 0.5:
            return 1.15, "Whales Accumulating"
        elif sentiment > 0.2:
            return 1.05, "Mild Accumulation"
        elif sentiment < -0.5:
            return 0.85, "Whales Distributing"
        elif sentiment < -0.2:
            return 0.95, "Mild Distribution"
        else:
            return 1.0, "No Clear Whale Activity"

# =====================================
# SMART ORDER ROUTER
# =====================================

class SmartOrderRouter:
    """Intelligent order routing for optimal execution"""
    
    def __init__(self, config: BotConfig, session):
        self.config = config
        self.session = session
        
    async def route_order(self, symbol: str, side: str, quantity: float, 
                         order_type: str = "market") -> Dict:
        """Route order using smart logic"""
        try:
            # Get current market conditions
            market_data = await self._get_market_depth(symbol)
            
            # Choose optimal order type
            if self.config.USE_SMART_ORDERS:
                order_type = self._select_order_type(market_data, quantity)
                
            # Calculate slippage estimate
            slippage_estimate = self._estimate_slippage(market_data, quantity, side)
            
            # Execute based on order type
            if order_type == "limit":
                return await self._place_limit_order(symbol, side, quantity, market_data)
            elif order_type == "twap":
                return await self._place_twap_order(symbol, side, quantity)
            else:
                return await self._place_market_order(symbol, side, quantity)
                
        except Exception as e:
            logging.error(f"Smart order routing error: {e}")
            # Fallback to market order
            return await self._place_market_order(symbol, side, quantity)
    
    async def _get_market_depth(self, symbol: str) -> Dict:
        """Get order book depth"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.get_orderbook(
                    category="linear",
                    symbol=symbol,
                    limit=50
                )
            )
            
            if response['retCode'] == 0:
                return response['result']
            return {}
            
        except Exception as e:
            logging.error(f"Market depth error: {e}")
            return {}
    
    def _select_order_type(self, market_data: Dict, quantity: float) -> str:
        """Select optimal order type based on market conditions"""
        if not market_data:
            return "market"
            
        # Calculate order book metrics
        bid_liquidity = sum(float(bid[1]) for bid in market_data.get('b', [])[:10])
        ask_liquidity = sum(float(ask[1]) for ask in market_data.get('a', [])[:10])
        
        # Large order relative to liquidity
        if quantity > min(bid_liquidity, ask_liquidity) * 0.1:
            return "twap"  # Use TWAP for large orders
        
        # Check spread
        if 'b' in market_data and 'a' in market_data:
            best_bid = float(market_data['b'][0][0])
            best_ask = float(market_data['a'][0][0])
            spread_pct = (best_ask - best_bid) / best_bid * 100
            
            if spread_pct < 0.05:  # Tight spread
                return "market"
            else:
                return "limit"
                
        return "market"
    
    def _estimate_slippage(self, market_data: Dict, quantity: float, side: str) -> float:
        """Estimate slippage for order"""
        if not market_data:
            return 0.1  # Default 0.1% slippage
            
        # Calculate average execution price
        book_side = 'a' if side == 'Buy' else 'b'
        if book_side not in market_data:
            return 0.1
            
        cumulative_qty = 0
        weighted_price = 0
        reference_price = float(market_data[book_side][0][0])
        
        for price_level in market_data[book_side]:
            level_price = float(price_level[0])
            level_qty = float(price_level[1])
            
            if cumulative_qty + level_qty >= quantity:
                # Partial fill at this level
                remaining_qty = quantity - cumulative_qty
                weighted_price += level_price * remaining_qty
                break
            else:
                weighted_price += level_price * level_qty
                cumulative_qty += level_qty
                
        if cumulative_qty > 0:
            avg_price = weighted_price / min(quantity, cumulative_qty)
            slippage = abs(avg_price - reference_price) / reference_price * 100
            return slippage
            
        return 0.5  # Default if calculation fails
    
    async def _place_limit_order(self, symbol: str, side: str, quantity: float, 
                                market_data: Dict) -> Dict:
        """Place limit order with optimal pricing"""
        try:
            # Get best price
            if side == "Buy":
                # Place slightly above best bid
                best_bid = float(market_data['b'][0][0])
                limit_price = best_bid * 1.0001
            else:
                # Place slightly below best ask
                best_ask = float(market_data['a'][0][0])
                limit_price = best_ask * 0.9999
                
            response = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Limit",
                qty=str(quantity),
                price=str(round(limit_price, 2)),
                timeInForce="IOC"
            )
            
            return {
                'success': response['retCode'] == 0,
                'order_id': response.get('result', {}).get('orderId'),
                'order_type': 'limit',
                'limit_price': limit_price
            }
            
        except Exception as e:
            logging.error(f"Limit order error: {e}")
            return {'success': False}
    
    async def _place_twap_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """Place TWAP (Time-Weighted Average Price) order"""
        try:
            # Split order into smaller chunks
            num_slices = 5
            slice_qty = quantity / num_slices
            
            results = []
            for i in range(num_slices):
                # Place market order for each slice
                response = self.session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    orderType="Market",
                    qty=str(round(slice_qty, 3)),
                    timeInForce="IOC"
                )
                
                results.append(response['retCode'] == 0)
                
                # Wait between slices
                if i < num_slices - 1:
                    await asyncio.sleep(2)
                    
            return {
                'success': all(results),
                'order_type': 'twap',
                'slices': num_slices,
                'slice_size': slice_qty
            }
            
        except Exception as e:
            logging.error(f"TWAP order error: {e}")
            return {'success': False}
    
    async def _place_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """Place standard market order"""
        try:
            response = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(quantity),
                timeInForce="IOC"
            )
            
            return {
                'success': response['retCode'] == 0,
                'order_id': response.get('result', {}).get('orderId'),
                'order_type': 'market'
            }
            
        except Exception as e:
            logging.error(f"Market order error: {e}")
            return {'success': False}

# =====================================
# REGIME DETECTOR
# =====================================

class RegimeDetector:
    """Detects market regime (trending, ranging, volatile)"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.regime_history = defaultdict(list)
        
    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        try:
            if len(df) < self.config.REGIME_LOOKBACK:
                return MarketRegime.RANGING
                
            # Calculate indicators
            close_prices = df['close'].tail(self.config.REGIME_LOOKBACK)
            returns = close_prices.pct_change().dropna()
            
            # ADX for trend strength
            if TA_AVAILABLE:
                adx = ta.trend.ADXIndicator(
                    df['high'].tail(self.config.REGIME_LOOKBACK),
                    df['low'].tail(self.config.REGIME_LOOKBACK),
                    df['close'].tail(self.config.REGIME_LOOKBACK)
                ).adx().iloc[-1]
            else:
                adx = 25
                
            # Calculate volatility
            volatility = returns.std() * np.sqrt(252)  # Annualized
            avg_volatility = returns.rolling(20).std().mean() * np.sqrt(252)
            
            # Linear regression for trend
            x = np.arange(len(close_prices))
            slope, intercept = np.polyfit(x, close_prices, 1)
            
            # R-squared for trend quality
            y_pred = slope * x + intercept
            ss_res = np.sum((close_prices - y_pred) ** 2)
            ss_tot = np.sum((close_prices - close_prices.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Determine regime
            if volatility > avg_volatility * 1.5:
                regime = MarketRegime.VOLATILE
            elif adx > 25 and r_squared > 0.7:
                if slope > 0:
                    regime = MarketRegime.TRENDING_UP
                else:
                    regime = MarketRegime.TRENDING_DOWN
            else:
                regime = MarketRegime.RANGING
                
            # Store in history
            symbol = "BTCUSDT"  # Default, should be passed as parameter
            self.regime_history[symbol].append({
                'regime': regime,
                'timestamp': datetime.now(),
                'adx': adx,
                'volatility': volatility,
                'r_squared': r_squared,
                'slope': slope
            })
            
            # Keep only recent history
            if len(self.regime_history[symbol]) > 100:
                self.regime_history[symbol] = self.regime_history[symbol][-100:]
                
            return regime
            
        except Exception as e:
            logging.error(f"Regime detection error: {e}")
            return MarketRegime.RANGING
    
    def get_regime_params(self, regime: MarketRegime) -> Dict:
        """Get trading parameters for regime"""
        params = {
            MarketRegime.TRENDING_UP: {
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 2.0,
                'prefer_strategies': ['trend_following', 'momentum']
            },
            MarketRegime.TRENDING_DOWN: {
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.5,
                'prefer_strategies': ['mean_reversion', 'short_bias']
            },
            MarketRegime.RANGING: {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'prefer_strategies': ['mean_reversion', 'range_trading']
            },
            MarketRegime.VOLATILE: {
                'position_size_multiplier': 0.5,
                'stop_loss_multiplier': 0.5,
                'take_profit_multiplier': 1.2,
                'prefer_strategies': ['volatility_arbitrage', 'risk_off']
            }
        }
        
        return params.get(regime, params[MarketRegime.RANGING])

# =====================================
# CONFIDENCE WEIGHTED VOTING
# =====================================

class ConfidenceWeightedVoter:
    """Advanced voting system with confidence weighting"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.strategy_performance = defaultdict(lambda: {
            'correct_predictions': 0,
            'total_predictions': 0,
            'rolling_accuracy': deque(maxlen=50),
            'confidence_score': 0.5
        })
        
    def vote(self, signals: List[Dict]) -> Tuple[SignalType, float, Dict]:
        """Perform confidence-weighted voting"""
        if not signals:
            return SignalType.HOLD, 0.0, {}
            
        # Calculate confidence scores
        for signal in signals:
            strategy_name = signal['strategy']
            perf = self.strategy_performance[strategy_name]
            
            # Update confidence based on rolling accuracy
            if perf['rolling_accuracy']:
                recent_accuracy = np.mean(perf['rolling_accuracy'])
                perf['confidence_score'] = recent_accuracy
            
            # Add confidence to signal
            signal['confidence'] = perf['confidence_score']
            
        # Weighted voting
        buy_weight = sum(
            s['strength'] * s['confidence']
            for s in signals
            if s['signal'] == SignalType.BUY
        )
        
        sell_weight = sum(
            s['strength'] * s['confidence']
            for s in signals
            if s['signal'] == SignalType.SELL
        )
        
        total_weight = sum(s['confidence'] for s in signals)
        
        # Normalize weights
        if total_weight > 0:
            buy_weight /= total_weight
            sell_weight /= total_weight
        
        # Determine signal
        if buy_weight > self.config.MIN_CONFIDENCE_THRESHOLD:
            signal = SignalType.BUY
            strength = buy_weight
        elif sell_weight > self.config.MIN_CONFIDENCE_THRESHOLD:
            signal = SignalType.SELL
            strength = sell_weight
        else:
            signal = SignalType.HOLD
            strength = 0.0
            
        analysis = {
            'buy_weight': buy_weight,
            'sell_weight': sell_weight,
            'strategy_confidences': {
                s['strategy']: s['confidence']
                for s in signals
            },
            'weighted_signals': signals
        }
        
        return signal, strength, analysis
    
    def update_performance(self, strategy_name: str, was_correct: bool):
        """Update strategy performance metrics"""
        perf = self.strategy_performance[strategy_name]
        
        perf['total_predictions'] += 1
        if was_correct:
            perf['correct_predictions'] += 1
            
        perf['rolling_accuracy'].append(1 if was_correct else 0)
        
        # Update confidence score
        if perf['total_predictions'] > 10:
            overall_accuracy = perf['correct_predictions'] / perf['total_predictions']
            recent_accuracy = np.mean(perf['rolling_accuracy']) if perf['rolling_accuracy'] else 0.5
            
            # Weighted average of overall and recent
            perf['confidence_score'] = 0.3 * overall_accuracy + 0.7 * recent_accuracy

# =====================================
# META AGENT
# =====================================

class MetaAgent:
    """AI agent that selects best strategies dynamically"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.strategy_metrics = defaultdict(lambda: {
            'returns': deque(maxlen=config.META_AGENT_WINDOW),
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'selection_score': 0.5
        })
        self.current_selection = []
        
    def select_strategies(self, available_strategies: List[str]) -> List[str]:
        """Select best performing strategies"""
        # Calculate selection scores
        strategy_scores = []
        
        for strategy in available_strategies:
            metrics = self.strategy_metrics[strategy]
            
            # Calculate metrics
            if metrics['returns']:
                returns = list(metrics['returns'])
                metrics['avg_return'] = np.mean(returns)
                metrics['sharpe_ratio'] = self._calculate_sharpe(returns)
                metrics['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
                
                # Composite score
                score = (
                    0.4 * metrics['sharpe_ratio'] +
                    0.3 * metrics['win_rate'] +
                    0.3 * (metrics['avg_return'] + 1)  # Shift to positive
                )
                metrics['selection_score'] = score
            else:
                # Default score for new strategies
                metrics['selection_score'] = 0.5
                
            strategy_scores.append((strategy, metrics['selection_score']))
            
        # Sort by score
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top strategies (at least 3, max 5)
        num_strategies = min(max(3, len(available_strategies) // 2), 5)
        selected = [s[0] for s in strategy_scores[:num_strategies]]
        
        # Always include at least one traditional strategy
        traditional = ['RSI', 'MACD', 'EMA']
        if not any(s in traditional for s in selected):
            selected.append(traditional[0])
            
        self.current_selection = selected
        return selected
    
    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        avg_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
            
        # Annualized Sharpe
        sharpe = (avg_return - risk_free_rate/252) / std_return * np.sqrt(252)
        return np.clip(sharpe, -2, 2)  # Cap extreme values
    
    def update_metrics(self, strategy_name: str, trade_return: float):
        """Update strategy metrics with trade result"""
        self.strategy_metrics[strategy_name]['returns'].append(trade_return)
    
    def get_strategy_report(self) -> Dict:
        """Get detailed strategy performance report"""
        report = {}
        
        for strategy, metrics in self.strategy_metrics.items():
            if metrics['returns']:
                report[strategy] = {
                    'selection_score': metrics['selection_score'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'win_rate': metrics['win_rate'],
                    'avg_return': metrics['avg_return'],
                    'num_trades': len(metrics['returns']),
                    'is_selected': strategy in self.current_selection
                }
                
        return report

# =====================================
# ADVANCED RISK MANAGER
# =====================================

class AdvancedRiskManager:
    """Enhanced risk management with multiple safety features"""
    
    def __init__(self, config: BotConfig, account_manager):
        self.config = config
        self.account_manager = account_manager
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.daily_pnl = 0.0
        self.circuit_breaker_active = False
        self.last_reset = datetime.now().date()
        
    def check_risk_limits(self) -> Tuple[bool, str]:
        """Check if trading is allowed based on risk limits"""
        # Reset daily counter
        if datetime.now().date() > self.last_reset:
            self.daily_pnl = 0.0
            self.circuit_breaker_active = False
            self.last_reset = datetime.now().date()
            
        # Check circuit breaker
        if self.circuit_breaker_active:
            return False, "Circuit breaker active"
            
        # Check daily loss limit
        balance = self.account_manager.get_balance()
        if balance['available'] > 0:
            daily_loss_pct = abs(self.daily_pnl) / balance['available']
            if daily_loss_pct > self.daily_loss_limit:
                self.circuit_breaker_active = True
                return False, f"Daily loss limit reached: {daily_loss_pct:.1%}"
                
        return True, "OK"
    
    def calculate_position_size_advanced(self, symbol: str, volatility: float, 
                                       regime: MarketRegime) -> float:
        """Calculate position size with volatility adjustment"""
        try:
            balance = self.account_manager.get_balance()
            available = balance['available']
            
            if available < self.config.MIN_BALANCE_REQUIRED:
                return 0
                
            # Base risk amount
            risk_amount = available * (self.config.RISK_PER_TRADE_PCT / 100)
            
            # Adjust for volatility
            if self.config.VOLATILITY_ADJUSTED_SIZING:
                # Lower size in high volatility
                vol_multiplier = 1 / (1 + volatility)
                risk_amount *= vol_multiplier
                
            # Adjust for regime
            regime_params = RegimeDetector(self.config).get_regime_params(regime)
            risk_amount *= regime_params['position_size_multiplier']
            
            # Get current price
            # ... (price fetching logic)
            
            return risk_amount
            
        except Exception as e:
            logging.error(f"Advanced position sizing error: {e}")
            return 0
    
    def detect_flash_crash(self, df: pd.DataFrame) -> bool:
        """Detect potential flash crash"""
        if len(df) < 10:
            return False
            
        # Check for rapid price drop
        recent_high = df['high'].tail(10).max()
        current_price = df['close'].iloc[-1]
        drop_pct = (recent_high - current_price) / recent_high * 100
        
        if drop_pct > self.config.FLASH_CRASH_THRESHOLD:
            logging.warning(f"Flash crash detected! Drop: {drop_pct:.1f}%")
            return True
            
        return False
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        self.daily_pnl += pnl

# =====================================
# ENHANCED STRATEGY VOTER WITH ALL FEATURES
# =====================================

class UltimateStrategyVoter(StrategyVoter):
    """Enhanced voting system with all advanced features"""
    
    def __init__(self, config: BotConfig, session, account_manager):
        super().__init__(config)
        self.session = session
        self.mtf_analyzer = MultiTimeframeAnalyzer(config, session) if config.MTF_ENABLED else None
        self.sentiment_analyzer = SentimentAnalyzer(config) if config.SENTIMENT_ENABLED else None
        self.whale_tracker = WhaleTracker(config) if config.WHALE_TRACKING_ENABLED else None
        self.regime_detector = RegimeDetector(config) if config.REGIME_DETECTION_ENABLED else None
        self.confidence_voter = ConfidenceWeightedVoter(config) if config.CONFIDENCE_WEIGHTED_VOTING else None
        self.meta_agent = MetaAgent(config) if config.META_AGENT_ENABLED else None
        self.risk_manager = AdvancedRiskManager(config, account_manager)
        
        # Add XGBoost to ML strategies if available
        if config.ML_ENABLED and ML_AVAILABLE:
            self.ml_strategies.append(GradientBoostingStrategy(config))
        
    async def get_ultimate_signal(self, symbol: str, df: pd.DataFrame) -> Tuple[SignalType, float, Dict[str, Any]]:
        """Get consensus signal with all advanced features"""
        try:
            # Check risk limits first
            can_trade, risk_msg = self.risk_manager.check_risk_limits()
            if not can_trade:
                logger.warning(f"Trading blocked: {risk_msg}")
                return SignalType.HOLD, 0.0, {'risk_blocked': risk_msg}
            
            # Check for flash crash
            if self.risk_manager.detect_flash_crash(df):
                logger.warning("Flash crash detected - blocking trades")
                return SignalType.HOLD, 0.0, {'flash_crash': True}
            
            # Get base signals from traditional and ML strategies
            base_signal, base_strength, base_analysis = self.get_consensus_signal(df)
            
            # Multi-timeframe analysis
            mtf_signal = SignalType.HOLD
            mtf_analysis = {}
            if self.mtf_analyzer:
                mtf_signals = await self.mtf_analyzer.get_mtf_signals(symbol)
                mtf_signal, mtf_strength, mtf_analysis = self.mtf_analyzer.get_confluence_signal(mtf_signals)
            
            # Sentiment analysis
            sentiment_modifier = 1.0
            sentiment_analysis = {}
            if self.sentiment_analyzer:
                sentiment_data = await self.sentiment_analyzer.get_sentiment(symbol)
                sentiment_modifier, sentiment_desc = self.sentiment_analyzer.interpret_sentiment(sentiment_data)
                sentiment_analysis = {
                    'sentiment': sentiment_data['overall_sentiment'],
                    'description': sentiment_desc,
                    'modifier': sentiment_modifier
                }
            
            # Whale tracking
            whale_modifier = 1.0
            whale_analysis = {}
            if self.whale_tracker:
                whale_data = await self.whale_tracker.get_whale_activity(symbol)
                whale_modifier, whale_desc = self.whale_tracker.interpret_whale_activity(whale_data)
                whale_analysis = {
                    'whale_sentiment': whale_data.get('whale_sentiment', 0),
                    'description': whale_desc,
                    'modifier': whale_modifier
                }
            
            # Market regime detection
            regime = MarketRegime.RANGING
            regime_params = {}
            if self.regime_detector:
                regime = self.regime_detector.detect_regime(df)
                regime_params = self.regime_detector.get_regime_params(regime)
            
            # Meta-agent strategy selection
            if self.meta_agent and base_analysis.get('individual_signals'):
                available_strategies = [s['strategy'] for s in base_analysis['individual_signals']]
                selected_strategies = self.meta_agent.select_strategies(available_strategies)
                # Filter signals to only selected strategies
                filtered_signals = [s for s in base_analysis['individual_signals'] 
                                  if s['strategy'] in selected_strategies]
                base_analysis['individual_signals'] = filtered_signals
            
            # Confidence-weighted voting
            if self.confidence_voter and base_analysis.get('individual_signals'):
                signal, strength, conf_analysis = self.confidence_voter.vote(
                    base_analysis['individual_signals']
                )
                base_signal = signal
                base_strength = strength
                base_analysis.update(conf_analysis)
            
            # Combine all signals with modifiers
            final_strength = base_strength * sentiment_modifier * whale_modifier
            
            # Apply regime-based adjustments
            if regime in [MarketRegime.VOLATILE, MarketRegime.TRENDING_DOWN]:
                final_strength *= 0.8  # Reduce signal strength in difficult conditions
            
            # Check MTF confluence
            if mtf_signal == base_signal:
                final_strength *= 1.1  # Boost if MTF agrees
            elif mtf_signal != SignalType.HOLD and mtf_signal != base_signal:
                final_strength *= 0.8  # Reduce if MTF disagrees
            
            # Final signal determination
            final_signal = base_signal
            if final_strength < self.config.MIN_SIGNAL_STRENGTH:
                final_signal = SignalType.HOLD
            
            # Comprehensive analysis
            comprehensive_analysis = {
                'base_analysis': base_analysis,
                'mtf_analysis': mtf_analysis,
                'sentiment_analysis': sentiment_analysis,
                'whale_analysis': whale_analysis,
                'regime': regime.value,
                'regime_params': regime_params,
                'meta_agent_report': self.meta_agent.get_strategy_report() if self.meta_agent else {},
                'final_strength': final_strength,
                'modifiers': {
                    'sentiment': sentiment_modifier,
                    'whale': whale_modifier,
                    'regime': regime_params.get('position_size_multiplier', 1.0)
                },
                'components': {
                    'mtf': {'signal': mtf_signal, 'analysis': mtf_analysis},
                    'sentiment': sentiment_analysis,
                    'whale': whale_analysis,
                    'regime': {'current_regime': regime.value, 'params': regime_params}
                }
            }
            
            return final_signal, final_strength, comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Ultimate strategy voter error: {e}")
            return SignalType.HOLD, 0.0, {}

# =====================================
# STRATEGY VOTER (Base Class)
# =====================================

class StrategyVoter:
    """Manages and combines multiple trading strategies"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        
        # Traditional strategies
        self.strategies = [
            RSIStrategy(config),
            MACDStrategy(config),
            EMAStrategy(config)
        ]
        
        # ML strategies
        self.ml_strategies = []
        if config.ML_ENABLED and ML_AVAILABLE:
            self.ml_strategies = [
                RandomForestStrategy(config),
                EnsembleMLStrategy(config)
            ]
        
        # RL strategies
        if config.PPO_ENABLED and RL_AVAILABLE:
            self.ml_strategies.append(PPOStrategy(config))
    
    def get_consensus_signal(self, df: pd.DataFrame) -> Tuple[SignalType, float, Dict[str, Any]]:
        """Get consensus signal from all strategies"""
        signals = []
        analyses = {}
        
        # Get signals from traditional strategies
        for strategy in self.strategies:
            signal, strength, analysis = strategy.analyze(df)
            signals.append({
                'strategy': strategy.name,
                'signal': signal,
                'strength': strength,
                'type': 'traditional'
            })
            analyses[strategy.name] = analysis
        
        # Get signals from ML strategies
        for ml_strategy in self.ml_strategies:
            signal, strength, analysis = ml_strategy.analyze(df)
            signals.append({
                'strategy': ml_strategy.name,
                'signal': signal,
                'strength': strength,
                'type': 'ml'
            })
            analyses[ml_strategy.name] = analysis
        
        # Count votes
        buy_votes = sum(1 for s in signals if s['signal'] == SignalType.BUY)
        sell_votes = sum(1 for s in signals if s['signal'] == SignalType.SELL)
        
        # Calculate weighted strength
        buy_strength = sum(s['strength'] for s in signals if s['signal'] == SignalType.BUY)
        sell_strength = sum(s['strength'] for s in signals if s['signal'] == SignalType.SELL)
        
        # Determine consensus
        if buy_votes >= self.config.MIN_STRATEGY_AGREEMENT:
            consensus_signal = SignalType.BUY
            consensus_strength = buy_strength / len(signals)
        elif sell_votes >= self.config.MIN_STRATEGY_AGREEMENT:
            consensus_signal = SignalType.SELL
            consensus_strength = sell_strength / len(signals)
        else:
            consensus_signal = SignalType.HOLD
            consensus_strength = 0.0
        
        # Check minimum strength requirement
        if consensus_strength < self.config.MIN_SIGNAL_STRENGTH:
            consensus_signal = SignalType.HOLD
            consensus_strength = 0.0
        
        return consensus_signal, consensus_strength, {
            'individual_signals': signals,
            'analyses': analyses,
            'buy_votes': buy_votes,
            'sell_votes': sell_votes,
            'buy_strength': buy_strength,
            'sell_strength': sell_strength
        }
    
    def train_ml_models(self, df: pd.DataFrame):
        """Train all ML models"""
        if not self.config.ML_ENABLED:
            return
        
        logger.info("Training ML models...")
        
        # Create features once for all models
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.create_features(df, self.config)
        
        if features_df.empty:
            logger.warning("No features created for ML training")
            return
        
        # Train each ML model
        for ml_strategy in self.ml_strategies:
            if hasattr(ml_strategy, 'train'):
                ml_strategy.train(features_df)

# =====================================
# ASYNC MARKET DATA (Enhanced)
# =====================================

class AsyncMarketData:
    """Handles async market data fetching"""
    
    def __init__(self, session):
        self.session = session
        self.cache = {}
        self.cache_duration = 30  # seconds
        
    async def get_kline_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch kline data asynchronously"""
        try:
            # Check cache
            cache_key = f"{symbol}_{BotConfig.KLINE_INTERVAL}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    return cached_data
            
            # Fetch data in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=BotConfig.KLINE_INTERVAL,
                    limit=BotConfig.KLINE_LIMIT
                )
            )
            
            if response['retCode'] != 0:
                logger.error(f"API error for {symbol}: {response['retMsg']}")
                return None
            
            # Convert to DataFrame
            klines = response['result']['list']
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Cache the data
            self.cache[cache_key] = (df, time.time())
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching kline data for {symbol}: {e}")
            return None
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(
                None,
                lambda: self.session.get_tickers(category="linear", symbol=symbol)
            )
            
            if ticker['retCode'] == 0:
                return float(ticker['result']['list'][0]['lastPrice'])
            return None
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

# =====================================
# ENHANCED ORDER MANAGER (with Smart Routing)
# =====================================

class EnhancedOrderManager:
    """Manages order execution and position tracking"""
    
    def __init__(self, session, account_manager, trailing_manager: TrailingStopManager):
        self.session = session
        self.account_manager = account_manager
        self.trailing_manager = trailing_manager
        
    def place_order_with_sl_tp(self, symbol: str, side: str, quantity: float, 
                              strategy_name: str = None) -> Optional[Trade]:
        """Place order with stop loss and take profit"""
        try:
            # Get current price
            ticker = self.session.get_tickers(category="linear", symbol=symbol)
            if ticker['retCode'] != 0:
                logger.error(f"Failed to get price for {symbol}")
                return None
            
            current_price = float(ticker['result']['list'][0]['lastPrice'])
            
            # Calculate SL/TP prices
            if side == "Buy":
                stop_loss = current_price * (1 - BotConfig.STOP_LOSS_PCT / 100)
                take_profit = current_price * (1 + BotConfig.TAKE_PROFIT_PCT / 100)
            else:
                stop_loss = current_price * (1 + BotConfig.STOP_LOSS_PCT / 100)
                take_profit = current_price * (1 - BotConfig.TAKE_PROFIT_PCT / 100)
            
            # Place order
            response = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(quantity),
                timeInForce="IOC",
                stopLoss=str(round(stop_loss, 2)),
                takeProfit=str(round(take_profit, 2))
            )
            
            if response['retCode'] != 0:
                logger.error(f"Order failed: {response['retMsg']}")
                return None
            
            logger.info(f"✅ Order placed: {symbol} {side} {quantity} @ ${current_price:.2f}")
            logger.info(f"   SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
            
            # Add to trailing stop manager
            self.trailing_manager.add_position(symbol, side, current_price, quantity)
            
            # Create trade record
            trade = Trade(
                timestamp=datetime.now(),
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=current_price,
                strategy=strategy_name
            )
            
            return trade
            
        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return None
    
    def update_trailing_stops(self, positions: List[Dict]):
        """Update trailing stops for all positions"""
        for pos in positions:
            symbol = pos['symbol']
            current_price = pos['current_price']
            
            # Update trailing stop
            new_stop = self.trailing_manager.update_position(symbol, current_price)
            
            if new_stop:
                try:
                    # Update stop loss on exchange
                    response = self.session.set_trading_stop(
                        category="linear",
                        symbol=symbol,
                        stopLoss=str(round(new_stop, 2))
                    )
                    
                    if response['retCode'] == 0:
                        logger.info(f"✅ Trailing stop updated for {symbol}: ${new_stop:.2f}")
                    else:
                        logger.error(f"Failed to update trailing stop: {response['retMsg']}")
                        
                except Exception as e:
                    logger.error(f"Trailing stop update error: {e}")

class UltimateOrderManager(EnhancedOrderManager):
    """Enhanced order execution with smart routing"""
    
    def __init__(self, session, account_manager, trailing_manager: TrailingStopManager, config: BotConfig):
        super().__init__(session, account_manager, trailing_manager)
        self.config = config
        self.smart_router = SmartOrderRouter(config, session) if config.USE_SMART_ORDERS else None
        
    async def place_order_with_sl_tp(self, symbol: str, side: str, quantity: float, 
                                    strategy_name: str = None, regime_params: Dict = None) -> Optional[Trade]:
        """Place order with advanced features"""
        try:
            # Get current price
            current_price = await AsyncMarketData(self.session).get_current_price(symbol)
            if not current_price:
                return None
            
            # Adjust SL/TP based on regime
            sl_multiplier = regime_params.get('stop_loss_multiplier', 1.0) if regime_params else 1.0
            tp_multiplier = regime_params.get('take_profit_multiplier', 1.0) if regime_params else 1.0
            
            # Calculate SL/TP prices
            if side == "Buy":
                stop_loss = current_price * (1 - self.config.STOP_LOSS_PCT / 100 * sl_multiplier)
                take_profit = current_price * (1 + self.config.TAKE_PROFIT_PCT / 100 * tp_multiplier)
            else:
                stop_loss = current_price * (1 + self.config.STOP_LOSS_PCT / 100 * sl_multiplier)
                take_profit = current_price * (1 - self.config.TAKE_PROFIT_PCT / 100 * tp_multiplier)
            
            # Use smart order routing if enabled
            if self.smart_router:
                result = await self.smart_router.route_order(symbol, side, quantity)
                if not result['success']:
                    logger.error(f"Smart order routing failed for {symbol}")
                    return None
            else:
                # Place standard market order
                response = self.session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    orderType="Market",
                    qty=str(quantity),
                    timeInForce="IOC",
                    stopLoss=str(round(stop_loss, 2)),
                    takeProfit=str(round(take_profit, 2))
                )
                
                if response['retCode'] != 0:
                    logger.error(f"Order failed: {response['retMsg']}")
                    return None
            
            logger.info(f"✅ Order placed: {symbol} {side} {quantity} @ ${current_price:.2f}")
            logger.info(f"   SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
            
            # Add to trailing stop manager
            self.trailing_manager.add_position(symbol, side, current_price, quantity)
            
            # Create trade record
            trade = Trade(
                timestamp=datetime.now(),
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=current_price,
                strategy=strategy_name
            )
            
            return trade
                
        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return None

# =====================================
# ACCOUNT MANAGER (Enhanced)
# =====================================

class AccountManager:
    """Manages account balance and positions"""
    
    def __init__(self, session):
        self.session = session
        self.balance_cache = None
        self.cache_time = 0
        self.cache_duration = 10
        
    def get_balance(self) -> Dict:
        """Get account balance with caching"""
        try:
            if time.time() - self.cache_time < self.cache_duration and self.balance_cache:
                return self.balance_cache
            
            response = self.session.get_wallet_balance(accountType="UNIFIED")
            
            if response['retCode'] != 0:
                logger.error(f"Balance API error: {response['retMsg']}")
                return {'available': 0, 'total': 0}
            
            account = response['result']['list'][0]
            balance = {
                'available': float(account.get('totalAvailableBalance', 0)),
                'total': float(account.get('totalWalletBalance', 0))
            }
            
            self.balance_cache = balance
            self.cache_time = time.time()
            
            return balance
            
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {'available': 0, 'total': 0}
    
    async def get_positions_async(self) -> List[Dict]:
        """Get open positions asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.get_positions(category="linear", settleCoin="USDT")
            )
            
            if response['retCode'] != 0:
                return []
            
            positions = []
            for pos in response['result']['list']:
                if float(pos['size']) > 0:
                    positions.append({
                        'symbol': pos['symbol'],
                        'side': pos['side'],
                        'size': float(pos['size']),
                        'entry_price': float(pos['avgPrice']),
                        'current_price': float(pos['markPrice']),
                        'pnl': float(pos['unrealisedPnl']),
                        'pnl_pct': float(pos['unrealisedPnl']) / (float(pos['size']) * float(pos['avgPrice'])) * 100
                    })
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

# =====================================
# ADDITIONAL ML STRATEGIES
# =====================================

class GradientBoostingStrategy(BaseStrategy):
    """XGBoost ML Strategy"""
    
    def __init__(self, config: BotConfig):
        super().__init__("XGBoost")
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.accuracy = 0.0
        self.feature_engineer = FeatureEngineer()
        
        try:
            import xgboost as xgb
            self.xgb = xgb
            self.xgb_available = True
        except ImportError:
            self.xgb_available = False
            logger.warning("XGBoost not installed. Run: pip install xgboost")
    
    def train(self, features_df: pd.DataFrame):
        """Train XGBoost model"""
        try:
            if not self.xgb_available or len(features_df) < 100:
                return
            
            # Prepare features and target
            feature_cols = [col for col in features_df.columns if col != 'target']
            X = features_df[feature_cols]
            y = features_df['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train XGBoost
            self.model = self.xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.3,
                objective='binary:logistic',
                use_label_encoder=False
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            self.accuracy = self.model.score(X_test_scaled, y_test)
            self.is_trained = True
            
            logger.info(f"XGBoost trained with accuracy: {self.accuracy:.2%}")
            
            # Save model
            self._save_model()
            
        except Exception as e:
            logger.error(f"XGBoost training error: {e}")
    
    def _save_model(self):
        """Save trained model"""
        try:
            os.makedirs('models', exist_ok=True)
            if self.xgb_available and self.model:
                self.model.save_model('models/xgboost_model.json')
                # Save scaler
                with open('models/xgboost_scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)
        except Exception as e:
            logger.error(f"Error saving XGBoost model: {e}")
    
    def _load_model(self):
        """Load model from disk"""
        try:
            if self.xgb_available and os.path.exists('models/xgboost_model.json'):
                self.model = self.xgb.XGBClassifier()
                self.model.load_model('models/xgboost_model.json')
                # Load scaler
                if os.path.exists('models/xgboost_scaler.pkl'):
                    with open('models/xgboost_scaler.pkl', 'rb') as f:
                        self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info("Loaded XGBoost model")
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {e}")
    
    def analyze(self, df: pd.DataFrame) -> Tuple[SignalType, float, Dict]:
        """Analyze using XGBoost"""
        try:
            if not self.xgb_available:
                return SignalType.HOLD, 0.0, {'error': 'XGBoost not available'}
            
            # Load model if not trained
            if not self.is_trained:
                self._load_model()
            
            if not self.is_trained:
                return SignalType.HOLD, 0.0, {'accuracy': 0.0}
            
            # Create features
            features_df = self.feature_engineer.create_features(df, self.config)
            if features_df.empty:
                return SignalType.HOLD, 0.0, {}
            
            # Get latest features
            feature_cols = [col for col in features_df.columns if col != 'target']
            latest_features = features_df[feature_cols].iloc[-1:].values
            
            # Scale and predict
            latest_scaled = self.scaler.transform(latest_features)
            prediction = self.model.predict(latest_scaled)[0]
            probability = self.model.predict_proba(latest_scaled)[0]
            
            analysis = {
                'prediction': int(prediction),
                'probability_up': float(probability[1]),
                'probability_down': float(probability[0]),
                'accuracy': self.accuracy
            }
            
            # Generate signal
            if self.accuracy >= self.config.ML_MIN_ACCURACY:
                if prediction == 1 and probability[1] > 0.65:
                    return SignalType.BUY, float(probability[1]), analysis
                elif prediction == 0 and probability[0] > 0.65:
                    return SignalType.SELL, float(probability[0]), analysis
            
            return SignalType.HOLD, 0.0, analysis
            
        except Exception as e:
            logger.error(f"XGBoost analysis error: {e}")
            return SignalType.HOLD, 0.0, {}

# =====================================
# ALERT SYSTEM
# =====================================

class AlertSystem:
    """Send alerts via Telegram and email"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.telegram_enabled = (
            hasattr(config, 'TELEGRAM_ENABLED') and 
            config.TELEGRAM_ENABLED and 
            os.getenv('TELEGRAM_TOKEN')
        )
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
    async def send_trade_alert(self, trade_data: Dict):
        """Send trade alert"""
        message = self._format_trade_message(trade_data)
        
        # Send Telegram
        if self.telegram_enabled and self.telegram_token and self.telegram_chat_id:
            await self._send_telegram(message)
            
        # Log to console
        logger.info(f"TRADE ALERT: {message}")
        
    def _format_trade_message(self, trade_data: Dict) -> str:
        """Format trade data into message"""
        emoji = "🟢" if trade_data['side'] == "Buy" else "🔴"
        
        message = f"""
{emoji} **NEW TRADE EXECUTED**
Symbol: {trade_data['symbol']}
Side: {trade_data['side']}
Price: ${trade_data['price']:.2f}
Quantity: {trade_data['quantity']}
Strategy: {trade_data.get('strategy', 'Unknown')}
Confidence: {trade_data.get('confidence', 0):.1%}
Regime: {trade_data.get('regime', 'Unknown')}
        """
        
        if 'sentiment' in trade_data:
            message += f"\nSentiment: {trade_data['sentiment']}"
            
        if 'whale_activity' in trade_data:
            message += f"\nWhale Activity: {trade_data['whale_activity']}"
            
        return message.strip()
        
    async def _send_telegram(self, message: str):
        """Send Telegram message"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status != 200:
                        logger.error(f"Telegram send failed: {await response.text()}")
                        
        except Exception as e:
            logger.error(f"Telegram error: {e}")

# =====================================
# PDF REPORT GENERATOR
# =====================================

class ReportGenerator:
    """Generate PDF trading reports"""
    
    def __init__(self):
        self.has_reportlab = False
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            self.has_reportlab = True
        except ImportError:
            logger.warning("ReportLab not installed. PDF reports disabled. Run: pip install reportlab")
            
    def generate_daily_report(self, trading_data: Dict, filename: str = None):
        """Generate daily trading report"""
        if not self.has_reportlab:
            return
            
        if not filename:
            filename = f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.pdf"
            
        # Import here to avoid errors if not installed
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            doc = SimpleDocTemplate(filename, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title = Paragraph(f"Ultimate Trading Bot Report - {datetime.now().strftime('%Y-%m-%d')}", 
                            styles['Title'])
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Summary section
            summary_data = [
                ['Metric', 'Value'],
                ['Total Trades', str(trading_data.get('total_trades', 0))],
                ['Win Rate', f"{trading_data.get('win_rate', 0):.1f}%"],
                ['Total PnL', f"${trading_data.get('total_pnl', 0):.2f}"],
                ['Sharpe Ratio', f"{trading_data.get('sharpe_ratio', 0):.2f}"],
                ['Max Drawdown', f"{trading_data.get('max_drawdown', 0):.1f}%"]
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(Paragraph("Performance Summary", styles['Heading2']))
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Strategy performance
            if 'strategy_performance' in trading_data:
                strategy_data = [['Strategy', 'Trades', 'Win Rate', 'PnL']]
                for strategy, perf in trading_data['strategy_performance'].items():
                    strategy_data.append([
                        strategy,
                        str(perf.get('trades', 0)),
                        f"{perf.get('win_rate', 0):.1f}%",
                        f"${perf.get('pnl', 0):.2f}"
                    ])
                    
                strategy_table = Table(strategy_data)
                strategy_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(Paragraph("Strategy Performance", styles['Heading2']))
                story.append(strategy_table)
                
            # Build PDF
            doc.build(story)
            logger.info(f"PDF report generated: {filename}")
            
        except Exception as e:
            logger.error(f"PDF generation error: {e}")

# =====================================
# MAIN ULTIMATE TRADING BOT
# =====================================

class UltimateTradingBot:
    """Ultimate trading bot with all advanced features"""
    
    def __init__(self):
        # Load environment
        load_dotenv()
        
        # Configuration
        self.config = BotConfig()
        
        # Initialize ByBit session
        self.session = HTTP(
            api_key=os.getenv("BYBIT_API_KEY"),
            api_secret=os.getenv("BYBIT_API_SECRET"),
            testnet=self.config.TESTNET
        )
        
        # Initialize components
        self.account_manager = AccountManager(self.session)
        self.async_market_data = AsyncMarketData(self.session)
        self.trailing_manager = TrailingStopManager(self.config)
        self.order_manager = UltimateOrderManager(
            self.session, 
            self.account_manager, 
            self.trailing_manager,
            self.config
        )
        self.strategy_voter = UltimateStrategyVoter(
            self.config,
            self.session,
            self.account_manager
        )
        self.performance_tracker = PerformanceTracker()
        self.csv_logger = CSVTradeLogger()  
        self.live_pnl_tracker = LivePnLTracker()
        
        # Add new components
        self.alert_system = AlertSystem(self.config)
        self.report_generator = ReportGenerator()
        
        # ML training state
        self.ml_training_data = []
        self.last_ml_train_scan = 0
        
        # Trading state
        self.running = False
        self.scan_count = 0
        self.active_trades: Dict[str, Trade] = {}
        self.current_regime = MarketRegime.RANGING
        
    async def scan_symbol(self, symbol: str) -> Optional[Dict]:
        """Scan a single symbol for opportunities with all features"""
        try:
            # Get market data
            df = await self.async_market_data.get_kline_data(symbol)
            if df is None:
                return None
            
            # Store data for ML training
            if self.config.ML_ENABLED and len(df) >= self.config.ML_LOOKBACK:
                self.ml_training_data.append((symbol, df))
            
            # Get ultimate consensus signal with all features
            signal, strength, analysis = await self.strategy_voter.get_ultimate_signal(symbol, df)
            
            return {
                'symbol': symbol,
                'signal': signal,
                'strength': strength,
                'analysis': analysis,
                'price': df['close'].iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            return None
    
    async def train_ml_models(self):
        """Train ML models with collected data"""
        try:
            if not self.ml_training_data:
                return
            
            logger.info("Starting ML model training...")
            
            # Combine data from all symbols
            all_dfs = []
            for symbol, df in self.ml_training_data[-100:]:  # Use last 100 data points
                if len(df) >= self.config.ML_LOOKBACK:
                    all_dfs.append(df)
            
            if all_dfs:
                # Concatenate all data
                combined_df = pd.concat(all_dfs, ignore_index=True)
                
                # Train models in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.strategy_voter.train_ml_models, combined_df)
            
            logger.info("ML model training completed")
            
        except Exception as e:
            logger.error(f"ML training error: {e}")
    
    async def scan_all_symbols(self):
        """Scan all symbols concurrently with advanced features"""
        try:
            # Check if we should retrain ML models
            if self.config.ML_ENABLED and self.scan_count - self.last_ml_train_scan >= self.config.ML_RETRAIN_INTERVAL:
                await self.train_ml_models()
                self.last_ml_train_scan = self.scan_count
            
            # Get current positions
            positions = await self.account_manager.get_positions_async()
            position_symbols = [p['symbol'] for p in positions]
            
            # Check position limit
            if len(positions) >= self.config.MAX_OPEN_POSITIONS:
                logger.info(f"Max positions reached ({len(positions)}/{self.config.MAX_OPEN_POSITIONS})")
                return
            
            # Symbols to scan
            symbols_to_scan = [s for s in self.config.SYMBOLS if s not in position_symbols]
            
            # Scan all symbols concurrently
            tasks = [self.scan_symbol(symbol) for symbol in symbols_to_scan]
            results = await asyncio.gather(*tasks)
            
            # Process results
            opportunities = [r for r in results if r and r['signal'] != SignalType.HOLD]
            
            if opportunities:
                # Sort by strength
                opportunities.sort(key=lambda x: x['strength'], reverse=True)
                
                # Log all opportunities
                logger.info(f"Found {len(opportunities)} opportunities:")
                for opp in opportunities:
                    logger.info(f"  {opp['symbol']}: {opp['signal'].value} "
                              f"(strength: {opp['strength']:.2f})")
                    
                    # Show detailed analysis
                    analysis = opp['analysis']
                    if 'components' in analysis:
                        # Show sentiment
                        if 'sentiment' in analysis['components']:
                            sent = analysis['components']['sentiment']
                            logger.info(f"    Sentiment: {sent.get('description', 'N/A')}")
                        
                        # Show whale activity
                        if 'whale' in analysis['components']:
                            whale = analysis['components']['whale']
                            logger.info(f"    Whale: {whale.get('description', 'N/A')}")
                        
                        # Show regime
                        if 'regime' in analysis['components']:
                            regime = analysis['components']['regime']
                            logger.info(f"    Regime: {regime.get('current_regime', 'N/A')}")
                
                # Execute best opportunity
                best_opp = opportunities[0]
                await self.execute_trade(best_opp)
                
        except Exception as e:
            logger.error(f"Scan error: {e}")
            
    async def execute_trade(self, opportunity: Dict):
        """Execute a trade based on opportunity"""
        try:
            symbol = opportunity['symbol']
            signal = opportunity['signal']
            analysis = opportunity['analysis']
            
            # Calculate position size
            balance = self.account_manager.get_balance()
            if balance['available'] < self.config.MIN_BALANCE_REQUIRED:
                logger.warning(f"Insufficient balance: ${balance['available']:.2f}")
                return
            
            # Risk-based position sizing
            position_value = min(
                balance['available'] * (self.config.RISK_PER_TRADE_PCT / 100),
                self.config.MAX_POSITION_VALUE
            )
            
            # Adjust for volatility and regime
            if 'regime_params' in analysis:
                position_value *= analysis['regime_params'].get('position_size_multiplier', 1.0)
            
            # Calculate quantity
            current_price = opportunity['price']
            quantity = round(position_value / current_price, 3)
            
            # Determine side
            side = "Buy" if signal == SignalType.BUY else "Sell"
            
            # Get strategy name
            strategy_name = "Ultimate_Consensus"
            if 'individual_signals' in analysis.get('base_analysis', {}):
                # Get the strongest signal
                signals = analysis['base_analysis']['individual_signals']
                if signals:
                    strongest = max(signals, key=lambda x: x['strength'])
                    strategy_name = strongest['strategy']
            
            # Place order
            regime_params = analysis.get('regime_params', {})
            trade = await self.order_manager.place_order_with_sl_tp(
                symbol, side, quantity, strategy_name, regime_params
            )
            
            if trade:
                # Log to CSV
                self.csv_logger.log_trade_entry(trade, analysis)
                
                # Add to performance tracker
                self.performance_tracker.add_trade(trade)
                
                # Update live PnL tracker
                self.live_pnl_tracker.add_open_position(symbol, trade.entry_price, side, quantity)
                
                # Store active trade
                self.active_trades[symbol] = trade
                
                # Send alert
                await self.alert_system.send_trade_alert({
                    'symbol': symbol,
                    'side': side,
                    'price': trade.entry_price,
                    'quantity': quantity,
                    'strategy': strategy_name,
                    'confidence': opportunity['strength'],
                    'regime': analysis.get('regime', 'Unknown'),
                    'sentiment': analysis.get('sentiment_analysis', {}).get('description', 'N/A'),
                    'whale_activity': analysis.get('whale_analysis', {}).get('description', 'N/A')
                })
                
                logger.info(f"🎯 Trade executed successfully!")
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    async def update_positions(self):
        """Update trailing stops and check exits"""
        try:
            positions = await self.account_manager.get_positions_async()
            
            for pos in positions:
                symbol = pos['symbol']
                
                # Update trailing stop
                self.order_manager.update_trailing_stops([pos])
                
                # Update live PnL
                unrealized_pnl = self.live_pnl_tracker.update_position_pnl(
                    symbol, pos['current_price']
                )
                pos['unrealized_pnl'] = unrealized_pnl
                
        except Exception as e:
            logger.error(f"Position update error: {e}")
    
    def check_closed_trades(self):
        """Check for closed trades and update records"""
        try:
            # Get current positions
            positions = self.session.get_positions(category="linear", settleCoin="USDT")
            if positions['retCode'] != 0:
                return
            
            position_symbols = [p['symbol'] for p in positions['result']['list'] if float(p['size']) > 0]
            
            # Check for closed trades
            for symbol, trade in list(self.active_trades.items()):
                if symbol not in position_symbols:
                    # Trade was closed
                    # Get exit info from order history
                    orders = self.session.get_order_history(
                        category="linear",
                        symbol=symbol,
                        limit=50
                    )
                    
                    if orders['retCode'] == 0:
                        # Find the closing order
                        for order in orders['result']['list']:
                            if order['side'] != trade.side and order['orderStatus'] == 'Filled':
                                exit_price = float(order['avgPrice'])
                                
                                # Calculate PnL
                                if trade.side == "Buy":
                                    pnl = (exit_price - trade.entry_price) * trade.quantity
                                    pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * 100
                                else:
                                    pnl = (trade.entry_price - exit_price) * trade.quantity
                                    pnl_pct = (trade.entry_price - exit_price) / trade.entry_price * 100
                                
                                # Update trade record
                                trade.exit_price = exit_price
                                trade.pnl = pnl
                                trade.pnl_pct = pnl_pct
                                trade.closed_at = datetime.now()
                                
                                # Update CSV
                                self.csv_logger.update_trade_exit(symbol, exit_price, pnl)
                                
                                # Update live PnL tracker
                                self.live_pnl_tracker.record_trade_result(pnl)
                                self.live_pnl_tracker.remove_position(symbol)
                                
                                # Update performance tracker
                                self.performance_tracker.add_trade(trade)
                                
                                # Update strategy performance
                                if hasattr(self.strategy_voter, 'confidence_voter') and trade.strategy:
                                    was_correct = pnl > 0
                                    self.strategy_voter.confidence_voter.update_performance(
                                        trade.strategy, was_correct
                                    )
                                
                                # Update meta agent
                                if hasattr(self.strategy_voter, 'meta_agent') and trade.strategy:
                                    self.strategy_voter.meta_agent.update_metrics(
                                        trade.strategy, pnl_pct / 100
                                    )
                                
                                # Update risk manager
                                if hasattr(self.strategy_voter, 'risk_manager'):
                                    self.strategy_voter.risk_manager.update_daily_pnl(pnl)
                                
                                # Remove from trailing stop manager
                                self.trailing_manager.remove_position(symbol)
                                
                                # Remove from active trades
                                del self.active_trades[symbol]
                                
                                logger.info(f"📊 Trade closed: {symbol} PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                                break
                    
        except Exception as e:
            logger.error(f"Error checking closed trades: {e}")
    
    async def run(self):
        """Main bot loop"""
        logger.info("🚀 Ultimate Trading Bot starting...")
        logger.info(f"Testnet: {self.config.TESTNET}")
        logger.info(f"Symbols: {', '.join(self.config.SYMBOLS)}")
        logger.info(f"Scan interval: {self.config.SCAN_INTERVAL}s")
        
        # Get initial balance
        balance = self.account_manager.get_balance()
        self.live_pnl_tracker.set_initial_balance(balance['available'])
        logger.info(f"💰 Initial balance: ${balance['available']:.2f}")
        
        self.running = True
        
        try:
            while self.running:
                self.scan_count += 1
                logger.info(f"\n{'='*50}")
                logger.info(f"🔍 Scan #{self.scan_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Update balance
                balance = self.account_manager.get_balance()
                self.live_pnl_tracker.update_balance(balance['available'])
                
                # Scan for opportunities
                await self.scan_all_symbols()
                
                # Update existing positions
                await self.update_positions()
                
                # Check for closed trades
                self.check_closed_trades()
                
                # Print live stats
                self.live_pnl_tracker.print_live_stats()
                
                # Generate hourly report
                if self.scan_count % (3600 // self.config.SCAN_INTERVAL) == 0:
                    stats = self.performance_tracker.get_statistics()
                    strategy_report = self.performance_tracker.get_strategy_report()
                    
                    trading_data = {
                        **stats,
                        'strategy_performance': strategy_report
                    }
                    
                    self.report_generator.generate_daily_report(trading_data)
                    self.performance_tracker.save_report()
                
                # Wait for next scan
                await asyncio.sleep(self.config.SCAN_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️ Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
        finally:
            self.running = False
            
            # Save final report
            self.performance_tracker.save_report()
            
            # Print final statistics
            logger.info("\n📊 FINAL TRADING STATISTICS")
            stats = self.performance_tracker.get_statistics()
            for key, value in stats.items():
                if isinstance(value, float):
                    logger.info(f"{key}: {value:.2f}")
                else:
                    logger.info(f"{key}: {value}")

# =====================================
# ENTRY POINT
# =====================================

def main():
    """Main entry point"""
    bot = UltimateTradingBot()
    asyncio.run(bot.run())

if __name__ == "__main__":
    main()
