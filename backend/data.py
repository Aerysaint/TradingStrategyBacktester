import duckdb
import os
import pandas as pd

DATASETS = {
    "1minute": "NIFTY 50_minute.csv",
    "5minute": "NIFTY 50_5minute.csv",
    "15minute": "NIFTY 50_15minute.csv",
    "60minute": "NIFTY 50_60minute.csv",
    "day": "NIFTY 50_day.csv"
}

db_path = os.path.join(os.path.dirname(__file__), "market_data.duckdb")

def init_db():
    conn = duckdb.connect(db_path)
    tables = conn.execute("SHOW TABLES").fetchdf()
    existing_tables = tables['name'].values if not tables.empty else []
    
    for ds_name, csv_filename in DATASETS.items():
        table_name = f"data_{ds_name}"
        if table_name not in existing_tables:
            print(f"Initializing DuckDB with {csv_filename} data...")
            csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "datasets", csv_filename))
            if os.path.exists(csv_path):
                conn.execute(f"""
                    CREATE TABLE {table_name} AS 
                    SELECT 
                        TRY_CAST(date AS TIMESTAMP) as date,
                        CAST(open AS FLOAT) as open,
                        CAST(high AS FLOAT) as high,
                        CAST(low AS FLOAT) as low,
                        CAST(close AS FLOAT) as close,
                        CAST(volume AS BIGINT) as volume
                    FROM read_csv_auto('{csv_path}', header=True)
                    WHERE TRY_CAST(date AS TIMESTAMP) IS NOT NULL
                """)
                conn.execute(f"CREATE INDEX idx_{table_name}_date ON {table_name}(date)")
            else:
                print(f"Warning: {csv_path} not found.")

    print("DuckDB initialized.")
    return conn

# Singleton connection
conn = init_db()

def get_ohlcv_data(dataset: str = "1minute", start_date: str = None, end_date: str = None, limit: int = 5000):
    table_name = f"data_{dataset}"
    
    # Check if table exists
    tables = conn.execute("SHOW TABLES").fetchdf()
    if table_name not in tables['name'].values:
        return pd.DataFrame()
        
    query = f"SELECT * FROM {table_name}"
    conditions = []
    if start_date:
        conditions.append(f"date >= '{start_date}'")
    if end_date:
        conditions.append(f"date <= '{end_date}'")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += f" ORDER BY date ASC LIMIT {limit}"
    
    df = conn.execute(query).fetchdf()
    
    if not df.empty:
        df['date'] = df['date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    return df
