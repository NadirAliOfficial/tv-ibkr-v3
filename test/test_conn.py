"""Test IBKR/TWS connection directly"""
from ib_insync import IB
import asyncio

async def test_connection():
    ib = IB()
    
    try:
        print("Attempting to connect to TWS...")
        print("Host: 127.0.0.1")
        print("Port: 7497")
        print("Client ID: 1")
        
        await ib.connectAsync('127.0.0.1', 7497, clientId=1)
        
        print("✅ Connected successfully!")
        
        # Get account info
        account = ib.managedAccounts()[0]
        print(f"Account: {account}")
        
        # Get account values
        values = ib.accountValues()
        print(f"Account values loaded: {len(values)} items")
        
        # Disconnect
        ib.disconnect()
        print("✅ Disconnected successfully")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Is TWS running?")
        print("2. Go to File → Global Configuration → API → Settings")
        print("3. Enable 'Enable ActiveX and Socket Clients'")
        print("4. Check Socket port is 7497")
        print("5. Restart TWS after changes")

if __name__ == "__main__":
    asyncio.run(test_connection())