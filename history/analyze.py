import ccxt
import duckdb
import pandas as pd
import requests
from datetime import datetime, timezone

# --- Configuration ---
DB_PATH = '/opt/llm_trader/market_data.duckdb'
QWEN_URL = "http://localhost:8080/v1/chat/completions"
ASSETS = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'XRP', 'TRX', 'BNB', 'AVAX', 'LTC', 'XLM', 'APT', 'DOGE', 'SUI']
SYMBOLS = [f"{a}/USDT" for a in ASSETS]

exchange = ccxt.binance({'enableRateLimit': True})

def ingest_data():
    print(f"[{datetime.now()}] Ingesting market data...")
    try:
        tickers = exchange.fetch_tickers(SYMBOLS)
        data_list = []
        ts = datetime.now(timezone.utc)
        for symbol in SYMBOLS:
            t = tickers.get(symbol)
            if t:
                data_list.append({
                    'timestamp': ts,
                    'asset_id': symbol.split('/')[0],
                    'price_usd': t['last'],
                    'volume_24h': t.get('quoteVolume', 0.0),
                    'change_24h': t.get('percentage', 0.0),
                    'market_cap': 0.0
                })
        if data_list:
            df = pd.DataFrame(data_list)
            con = duckdb.connect(DB_PATH)
            con.execute("INSERT INTO market_history SELECT * FROM df")
            con.close()
    except Exception as e:
        print(f"Ingestion Error: {e}")

def get_anomaly_hotlist():
    con = duckdb.connect(DB_PATH)
    # Filter for Z-score < -2.0 (Buy zone) and require > 20 samples for statistical validity
    query = """
    WITH stats AS (
        SELECT a.asset_id || '/' || b.asset_id as pair, 
               AVG(a.price_usd / b.price_usd) as mean, 
               STDDEV(a.price_usd / b.price_usd) as std,
               COUNT(*) as sample_count
        FROM market_history a, market_history b
        WHERE a.asset_id != b.asset_id AND a.timestamp = b.timestamp
        GROUP BY pair HAVING sample_count > 20
    ),
    latest AS (
        SELECT a.asset_id || '/' || b.asset_id as pair, 
               (a.price_usd / b.price_usd) as ratio
        FROM market_history a, market_history b
        WHERE a.timestamp = (SELECT MAX(timestamp) FROM market_history)
        AND a.asset_id != b.asset_id
    )
    SELECT l.pair, l.ratio, s.mean, (l.ratio - s.mean) / NULLIF(s.std, 0) as z_score
    FROM latest l JOIN stats s ON l.pair = s.pair
    WHERE z_score < -2.0 
    ORDER BY z_score ASC LIMIT 5
    """
    df = con.execute(query).df()
    con.close()
    return df

def ask_qwen(anomaly_df):
    if anomaly_df.empty: return "Market stable. No high-conviction buy signals found."
    summary = anomaly_df[['pair', 'z_score']].to_string(index=False)
    prompt = (
        f"You are a growth-oriented spot trader. Analyze these oversold pairs (Z-score < -2.0):\n{summary}\n\n"
        "Identify the best growth opportunity. Provide a brief buy thesis based on the price ratio "
        "reversion to the mean. Do not suggest shorts or margin trades. Only suggest spot accumulation."
    )
    payload = {"model": "qwen-instruct", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    try:
        response = requests.post(QWEN_URL, json=payload, timeout=60)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"LLM Verdict Unavailable: {e}"

if __name__ == "__main__":
    ingest_data()
    anomalies = get_anomaly_hotlist()
    print("\n--- Top Growth Signals (Buy Zone) ---")
    print(anomalies[['pair', 'z_score']] if not anomalies.empty else "None")
    print("\n--- Qwen Verdict ---")
    print(ask_qwen(anomalies))
