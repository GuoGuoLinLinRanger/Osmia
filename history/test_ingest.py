import ccxt
import duckdb
import pandas as pd
from datetime import datetime, timezone

# --- Configuration ---
DB_PATH = '/opt/llm_trader/market_data.duckdb'
ASSETS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 'XRP/USDT', 
          'TRX/USDT', 'BNB/USDT', 'AVAX/USDT', 'LTC/USDT', 'XLM/USDT', 'APT/USDT', 
          'DOGE/USDT', 'SUI/USDT']

exchange = ccxt.binance({'enableRateLimit': True})

def ingest():
    print(f"[{datetime.now()}] Fetching market data...")
    try:
        tickers = exchange.fetch_tickers(ASSETS)
        data_list = []
        
        for symbol in ASSETS:
            t = tickers.get(symbol)
            if t:
                data_list.append({
                    'timestamp': datetime.now(timezone.utc),
                    'asset_id': symbol,
                    'price_usd': t['last'],
                    'volume_24h': t.get('quoteVolume', 0.0),
                    'change_24h': t.get('percentage', 0.0),
                    'market_cap': 0.0
                })
        
        if data_list:
            df = pd.DataFrame(data_list)
            con = duckdb.connect(DB_PATH)
            # Use 'INSERT INTO' with explicit column mapping for safety
            con.execute("INSERT INTO market_history SELECT * FROM df")
            con.close()
            print(f"Successfully ingested {len(data_list)} records.")
            
    except Exception as e:
        print(f"Ingestion failed: {e}")

if __name__ == "__main__":
    ingest()
