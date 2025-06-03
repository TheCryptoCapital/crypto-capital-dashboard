#!/usr/bin/env python3
"""
Quick test to check if your exchange API is working
"""
import requests
import json

def test_binance_connection():
    """Test basic Binance API connection"""
    print("🔍 Testing Binance API Connection...")
    
    try:
        # Test public endpoint (no API keys needed)
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '1m',
            'limit': 5
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS: Got {len(data)} candles")
            print("Sample data:")
            for i, candle in enumerate(data):
                timestamp, open_price, high, low, close, volume = candle[:6]
                print(f"  Candle {i+1}: Open=${float(open_price):,.2f}, Close=${float(close):,.2f}")
            return True
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_with_api_keys():
    """Test with your API keys (if you have them)"""
    print("\n🔍 Testing with API Keys...")
    
    # You'll need to add your actual API keys here
    API_KEY = "YOUR_API_KEY_HERE"
    SECRET_KEY = "YOUR_SECRET_KEY_HERE"
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️  No API keys provided - skipping authenticated test")
        return
    
    try:
        import hmac
        import hashlib
        import time
        
        timestamp = int(time.time() * 1000)
        
        # Test account info endpoint
        url = "https://api.binance.com/api/v3/account"
        query_string = f"timestamp={timestamp}"
        
        signature = hmac.new(
            SECRET_KEY.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            'X-MBX-APIKEY': API_KEY
        }
        
        response = requests.get(
            f"{url}?{query_string}&signature={signature}",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Keys working!")
            print(f"Account has {len(data.get('balances', []))} balance entries")
        else:
            print(f"❌ API Key test failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ API Key test error: {str(e)}")

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Exchange API Connection Test")
    print("=" * 50)
    
    # Test basic connection
    basic_works = test_binance_connection()
    
    # Test with API keys
    test_with_api_keys()
    
    print("\n" + "=" * 50)
    if basic_works:
        print("✅ Basic API connection is working!")
        print("🔧 The issue is likely in your bot's data fetching code.")
    else:
        print("❌ Can't connect to exchange API at all!")
        print("🔧 Check your internet connection and firewall settings.")
    print("=" * 50)
