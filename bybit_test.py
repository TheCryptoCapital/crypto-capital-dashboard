#!/usr/bin/env python3
"""
ByBit V5 API Connection Test
Test your V5 API credentials and data retrieval
"""

import os
import requests
import json
import time
import hmac
import hashlib
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Get credentials
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')
TESTNET = os.getenv('BYBIT_TESTNET', 'false').lower() == 'true'

print("=" * 60)
print("🔧 BYBIT V5 API CONNECTION TEST")
print("=" * 60)

print(f"API Key: {'*' * 8}...{API_KEY[-4:] if API_KEY else 'MISSING'}")
print(f"API Secret: {'*' * 8}...{API_SECRET[-4:] if API_SECRET else 'MISSING'}")
print(f"Testnet Mode: {TESTNET}")

# Determine base URL - V5 API
if TESTNET:
    BASE_URL = "https://api-testnet.bybit.com"
else:
    BASE_URL = "https://api.bybit.com"

print(f"Base URL: {BASE_URL}")

def generate_signature(method, endpoint, params, timestamp, recv_window="5000"):
    """Generate ByBit V5 API signature"""
    # For GET requests: timestamp + api_key + recv_window + query_string
    # For POST requests: timestamp + api_key + recv_window + raw_data
    
    if method.upper() == 'GET' and params:
        # Convert params to query string
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        param_str = str(timestamp) + API_KEY + recv_window + query_string
    elif method.upper() == 'POST' and params:
        # For POST, use JSON string
        param_str = str(timestamp) + API_KEY + recv_window + json.dumps(params, separators=(',', ':'))
    else:
        # No params
        param_str = str(timestamp) + API_KEY + recv_window
    
    return hmac.new(API_SECRET.encode('utf-8'), param_str.encode('utf-8'), hashlib.sha256).hexdigest()

def test_public_endpoint():
    """Test V5 public endpoint (no auth required)"""
    print("\n🔍 Testing V5 Public Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/v5/market/time", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ V5 Public API: SUCCESS")
            print(f"   Server Time: {data}")
            return True
        else:
            print(f"❌ V5 Public API: FAILED ({response.status_code})")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ V5 Public API: ERROR - {e}")
        return False

def test_kline_data():
    """Test V5 kline data (public endpoint)"""
    print("\n📊 Testing V5 Kline Data Retrieval...")
    try:
        params = {
            'category': 'spot',
            'symbol': 'BTCUSDT',
            'interval': '1',  # 1 minute
            'limit': 5
        }
        
        response = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('retCode') == 0:
                result = data.get('result', {}).get('list', [])
                print("✅ V5 Kline Data: SUCCESS")
                print(f"   Records Retrieved: {len(result)}")
                if result:
                    print(f"   Latest Price: {result[0][4] if len(result[0]) > 4 else 'N/A'}")
                return True
            else:
                print(f"❌ V5 Kline Data: API Error - {data.get('retMsg', 'Unknown')}")
                return False
        else:
            print(f"❌ V5 Kline Data: HTTP Error ({response.status_code})")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ V5 Kline Data: ERROR - {e}")
        return False

def test_private_endpoint():
    """Test V5 private endpoint (requires auth)"""
    print("\n🔐 Testing V5 Private Endpoint (Account Balance)...")
    
    if not API_KEY or not API_SECRET:
        print("❌ V5 Private API: No credentials provided")
        return False
        
    try:
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        
        params = {
            'accountType': 'UNIFIED'
        }
        
        # V5 API signature method for GET request
        signature = generate_signature('GET', '/v5/account/wallet-balance', params, timestamp, recv_window)
        
        headers = {
            'X-BAPI-API-KEY': API_KEY,
            'X-BAPI-SIGN': signature,
            'X-BAPI-SIGN-TYPE': '2',
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': recv_window,
            'Content-Type': 'application/json'
        }
        
        response = requests.get(f"{BASE_URL}/v5/account/wallet-balance", params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('retCode') == 0:
                print("✅ V5 Private API: SUCCESS")
                result = data.get('result', {})
                # Don't show actual balance for security
                print(f"   Account Access: Confirmed")
                print(f"   Account Type: {result.get('list', [{}])[0].get('accountType', 'Unknown') if result.get('list') else 'Unknown'}")
                return True
            else:
                print(f"❌ V5 Private API: API Error - {data.get('retMsg', 'Unknown')}")
                print(f"   Error Code: {data.get('retCode', 'Unknown')}")
                return False
        else:
            print(f"❌ V5 Private API: HTTP Error ({response.status_code})")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ V5 Private API: ERROR - {e}")
        return False

def main():
    print("\n🚀 Starting ByBit V5 API Tests...")
    
    results = {
        'public': test_public_endpoint(),
        'kline': test_kline_data(),
        'private': test_private_endpoint()
    }
    
    print("\n" + "=" * 60)
    print("📋 V5 API TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"V5 {test_name.capitalize()} API: {status}")
    
    if results['public'] and results['kline']:
        print("\n🎉 V5 API connectivity PERFECT! Market data works!")
        print("💡 Your bot just needs V5 endpoint updates in get_kline_data method.")
    elif not results['public']:
        print("\n⚠️  Network/connectivity issue to ByBit V5 servers.")
    elif not results['kline']:
        print("\n⚠️  V5 Kline data retrieval failing - check API endpoints.")
    
    if results['private']:
        print("🚀 Full V5 API access confirmed - ready for live trading!")
    elif not results['private'] and API_KEY and API_SECRET:
        print("⚠️  V5 Authentication needs signature fix - but market data works for trading!")

if __name__ == "__main__":
    main()
