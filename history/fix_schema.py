import duckdb

con = duckdb.connect('/opt/llm_trader/market_data.duckdb')

missing_columns = [
    ('vwap_ratio', 'DOUBLE'),
    ('volume_z', 'DOUBLE'),
    ('btc_corr', 'DOUBLE'),
    ('bid_depth', 'DOUBLE'),
    ('ask_depth', 'DOUBLE')
]

for col_name, col_type in missing_columns:
    try:
        con.execute(f"ALTER TABLE market_history ADD COLUMN {col_name} {col_type}")
        print(f"Added column: {col_name}")
    except Exception as e:
        print(f"Skipping {col_name} (already exists or error): {e}")

con.close()
