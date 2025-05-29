#!/usr/bin/env python3
import os
from pybit.unified_trading import HTTP

def test_api_connection():
    print("🔍 Testing Bybit API Connection...")
    
    session = HTTP(
        testnet=False,
        api_key=os.getenv('BYBIT_API_KEY'),
        api_secret=os.getenv('BYBIT_API_SECRET'),
    )
    
    try:
        print("1️⃣ Testing account access...")
        balance = session.get_wallet_balance(accountType="UNIFIED")
        print(f"✅ Account Balance Retrieved: {balance['retCode']}")
        
        print("2️⃣ Testing positions...")  
        positions = session.get_positions(category="linear", settleCoin="USDT")
        print(f"✅ Positions Retrieved: {positions['retCode']}")
        
        print("3️⃣ Testing orders...")
        orders = session.get_open_orders(category="linear")
        print(f"✅ Orders Retrieved: {orders['retCode']}")
        
        if balance['retCode'] == 0:
            total_balance = balance['result']['list'][0]['totalWalletBalance']
            available_balance = balance['result']['list'][0]['totalAvailableBalance']
            print(f"\n💰 Account Summary:")
            print(f"   Total Balance: ${total_balance}")
            print(f"   Available Balance: ${available_balance}")
            print("\n🎉 API FULLY CONNECTED FOR TRADING!")
            
    except Exception as e:
        print(f"❌ API Connection Failed: {e}")

if __name__ == "__main__":
    test_api_connection()
