import duckdb
import os

db_path = 'market_data.duckdb'
backup_path = 'market_history_backup.csv'

con = duckdb.connect(db_path)

# Export
con.execute(f"COPY market_history TO '{backup_path}' (HEADER, DELIMITER ',');")

# Drop
con.execute("DROP TABLE market_history;")

# Create New
con.execute("""
CREATE TABLE market_history (
    timestamp TIMESTAMPTZ NOT NULL,
    asset_id VARCHAR NOT NULL,
    price_usd DOUBLE,
    volume_24h DOUBLE,
    funding_rate DOUBLE,
    bid_depth DOUBLE,
    ask_depth DOUBLE,
    btc_corr DOUBLE,
    spread_pct DOUBLE,
    PRIMARY KEY (timestamp, asset_id)
);
""")

# Import
con.execute(f"COPY market_history FROM '{backup_path}' (HEADER, DELIMITER ',');")
con.close()
print("Migration successful.")
