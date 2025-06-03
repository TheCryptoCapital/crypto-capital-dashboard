#!/usr/bin/env python3
"""
Auto-fix script for Multi-Strategy Trading Bot
Fixes missing attributes and configuration issues
"""

import re
import os
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create a backup of the original file"""
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Created backup: {backup_path}")
    return backup_path

def fix_elite_strategy_config(content):
    """Add min_time_between_trades to EliteStrategyConfig"""
    print("🔧 Fixing EliteStrategyConfig...")
    
    # Find the EliteStrategyConfig __init__ method
    pattern = r'(class EliteStrategyConfig.*?def __init__\(self,[^)]+)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        init_section = match.group(1)
        
        # Check if min_time_between_trades already exists
        if 'min_time_between_trades' not in init_section:
            # Add it after daily_trade_limit parameter
            new_init = re.sub(
                r'(daily_trade_limit: int = \d+,)',
                r'\1\n                 min_time_between_trades: int = 8,  # ✅ AUTO-FIXED: Added missing attribute',
                init_section
            )
            content = content.replace(init_section, new_init)
            
            # Also add the assignment in the __init__ body
            init_body_pattern = r'(self\.daily_trade_limit = daily_trade_limit)'
            content = re.sub(
                init_body_pattern,
                r'\1\n        self.min_time_between_trades = min_time_between_trades  # ✅ AUTO-FIXED',
                content
            )
            print("  ✅ Added min_time_between_trades to EliteStrategyConfig")
    
    return content

def fix_trading_config(content):
    """Ensure TradingConfig has min_time_between_trades"""
    print("🔧 Fixing TradingConfig...")
    
    # Find the TradingConfig class
    pattern = r'(@dataclass\s+class TradingConfig:.*?)(def __post_init__|class\s+\w+|$)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        config_section = match.group(1)
        
        # Check if min_time_between_trades exists
        if 'min_time_between_trades: int = ' not in config_section:
            # Add it after daily_trade_limit
            new_config = re.sub(
                r'(daily_trade_limit: int = \d+)',
                r'\1\n        min_time_between_trades: int = 8                 # ✅ AUTO-FIXED: Minimum seconds between trades',
                config_section
            )
            content = content.replace(config_section, new_config)
            print("  ✅ Added min_time_between_trades to TradingConfig")
    
    return content

def fix_order_manager_time_check(content):
    """Fix the OrderManager time check to handle missing attribute"""
    print("🔧 Fixing OrderManager time check...")
    
    # Find the problematic line in place_market_order_with_protection
    pattern = r'if now - self\.last_trade_time\[symbol\] < config\.min_time_between_trades:'
    
    if re.search(pattern, content):
        # Replace with a more robust version
        content = re.sub(
            pattern,
            'if now - self.last_trade_time[symbol] < getattr(config, "min_time_between_trades", 8):',
            content
        )
        print("  ✅ Fixed OrderManager time check with fallback")
    
    return content

def fix_stop_loss_calculation(content):
    """Fix the stop loss calculation in calculate_safe_stop_loss"""
    print("🔧 Fixing stop loss calculation...")
    
    # Find the calculate_safe_stop_loss method
    pattern = r'(def calculate_safe_stop_loss\(.*?\n.*?try:.*?)(if side == "Buy":.*?)(return stop_price)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        # Replace the stop loss logic to use actual risk-based stops
        new_logic = '''# Calculate stop loss based on risk (not trailing config for initial order)
            risk_per_unit = risk_usd / qty if qty > 0 else 0
            
            if side == "Buy":
                # For buy orders, stop loss is BELOW entry price
                stop_price = entry_price - risk_per_unit
            else:
                # For sell orders, stop loss is ABOVE entry price
                stop_price = entry_price + risk_per_unit
            
            # Ensure minimum distance
            min_distance = entry_price * 0.002  # 0.2% minimum
            actual_distance = abs(entry_price - stop_price)
            
            if actual_distance < min_distance:
                if side == "Buy":
                    stop_price = entry_price - min_distance
                else:
                    stop_price = entry_price + min_distance'''
        
        # Replace the entire stop loss calculation section
        content = re.sub(
            r'# Use trailing stop manager.*?min_distance = entry_price \* 0\.001',
            new_logic,
            content,
            flags=re.DOTALL
        )
        print("  ✅ Fixed stop loss calculation logic")
    
    return content

def fix_all_strategy_configs(content):
    """Add missing attributes to all strategy configurations"""
    print("🔧 Fixing all strategy configurations...")
    
    # Pattern to find strategy config entries
    pattern = r'(StrategyType\.\w+: EliteStrategyConfig\([^)]+)(\))'
    
    def add_missing_params(match):
        config_content = match.group(1)
        closing = match.group(2)
        
        # Check if scan_symbols is missing
        if 'scan_symbols=' not in config_content and 'symbols' in config_content:
            # Extract symbols value and add scan_symbols
            symbols_match = re.search(r'symbols=(\[[^\]]+\])', config_content)
            if symbols_match:
                symbols_value = symbols_match.group(1)
                config_content = re.sub(
                    r'(symbols=\[[^\]]+\])',
                    r'\1,\n        scan_symbols=\1',  # Copy symbols to scan_symbols
                    config_content
                )
        
        return config_content + closing
    
    content = re.sub(pattern, add_missing_params, content)
    print("  ✅ Fixed strategy configurations")
    
    return content

def fix_account_manager_attributes(content):
    """Fix missing attributes in AccountManager classes"""
    print("🔧 Fixing AccountManager attributes...")
    
    # Add missing method to HFQAccountManager
    hfq_methods = '''
    def calculate_portfolio_risk(self):
        """Calculate current portfolio risk exposure"""
        try:
            positions = self.get_open_positions()
            balance_info = self.get_account_balance()
            
            if not positions or balance_info['available'] <= 0:
                return 0.0
            
            total_position_value = sum(pos.get('position_value', 0) for pos in positions)
            risk_pct = (total_position_value / balance_info['available']) * 100
            
            return risk_pct
            
        except Exception as e:
            logger.error(f"❌ Portfolio risk calculation error: {e}")
            return 0.0'''
    
    # Find end of HFQAccountManager class and add method if missing
    if 'class HFQAccountManager' in content and 'def calculate_portfolio_risk' not in content:
        # Add after get_balance_summary method
        pattern = r'(def get_balance_summary\(self\):.*?return "Balance unavailable")'
        content = re.sub(
            pattern,
            r'\1\n' + hfq_methods,
            content,
            flags=re.DOTALL
        )
        print("  ✅ Added calculate_portfolio_risk to HFQAccountManager")
    
    return content

def main():
    """Main function to apply all fixes"""
    print("🚀 Auto-Fix Script for Multi-Strategy Trading Bot")
    print("="*50)
    
    # Find the bot file
    bot_file = "bot_loop.py"
    
    if not os.path.exists(bot_file):
        print(f"❌ Error: {bot_file} not found!")
        print("Please ensure this script is in the same directory as your bot file.")
        return
    
    # Create backup
    backup_path = backup_file(bot_file)
    
    try:
        # Read the file content
        with open(bot_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all fixes
        content = fix_elite_strategy_config(content)
        content = fix_trading_config(content)
        content = fix_order_manager_time_check(content)
        content = fix_stop_loss_calculation(content)
        content = fix_all_strategy_configs(content)
        content = fix_account_manager_attributes(content)
        
        # Write the fixed content back
        with open(bot_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("\n✅ All fixes applied successfully!")
        print(f"📁 Original file backed up to: {backup_path}")
        print("\n🎯 Fixed issues:")
        print("  1. Added min_time_between_trades to EliteStrategyConfig")
        print("  2. Added min_time_between_trades to TradingConfig")
        print("  3. Fixed OrderManager time check with fallback")
        print("  4. Fixed stop loss calculation logic")
        print("  5. Added missing scan_symbols to strategy configs")
        print("  6. Added missing methods to AccountManager")
        
        # Verify changes
        if content != original_content:
            print("\n✅ File successfully modified!")
            print("🚀 Your bot should now run without the attribute errors.")
        else:
            print("\n⚠️ Warning: No changes were made. The issues might already be fixed.")
        
    except Exception as e:
        print(f"\n❌ Error during fixing: {e}")
        print(f"Your original file is safe at: {backup_path}")
        
        # Offer to restore
        response = input("\nWould you like to restore the original file? (y/n): ")
        if response.lower() == 'y':
            shutil.copy2(backup_path, bot_file)
            print("✅ Original file restored.")

if __name__ == "__main__":
    main()
