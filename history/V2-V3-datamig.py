#!/usr/bin/env python3
"""
migrate_db.py  —  one-shot migration from v0.2 schema to v0.3
Safe to run multiple times (checks before altering).
"""
import duckdb
import sys

DB_PATH = '/opt/llm_trader/market_data.duckdb'

def col_exists(con, table, col):
    df = con.execute(f"DESCRIBE {table}").df()
    return col in df['column_name'].values

def drop_col_if_exists(con, table, col):
    if col_exists(con, table, col):
        con.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
        print(f"  dropped  {table}.{col}")
    else:
        print(f"  skip     {table}.{col} (not found)")

def add_col_if_missing(con, table, col, typedef):
    if not col_exists(con, table, col):
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")
        print(f"  added    {table}.{col} {typedef}")
    else:
        print(f"  skip     {table}.{col} (already exists)")

def main():
    print(f"Connecting to {DB_PATH} …")
    con = duckdb.connect(DB_PATH)

    print("\n── market_history ──")
    # Remove dead NULL columns from prior version
    for col in ('funding_rate', 'bid_depth', 'ask_depth', 'btc_corr'):
        drop_col_if_exists(con, 'market_history', col)
    # Add missing columns
    add_col_if_missing(con, 'market_history', 'change_24h',   'DOUBLE')
    add_col_if_missing(con, 'market_history', 'high_low_pct', 'DOUBLE')

    print("\n── llm_verdicts ──")
    add_col_if_missing(con, 'llm_verdicts', 'prose', 'VARCHAR')

    print("\n── Final schemas ──")
    print("\nmarket_history:")
    print(con.execute("DESCRIBE market_history").df().to_string(index=False))
    print("\nllm_verdicts:")
    print(con.execute("DESCRIBE llm_verdicts").df().to_string(index=False))

    row_count = con.execute("SELECT COUNT(*) FROM market_history").fetchone()[0]
    verdict_count = con.execute("SELECT COUNT(*) FROM llm_verdicts").fetchone()[0]
    print(f"\n── Row counts ──")
    print(f"  market_history : {row_count}")
    print(f"  llm_verdicts   : {verdict_count}")

    con.close()
    print("\nMigration complete.")

if __name__ == "__main__":
    main()
