import sqlite3
import pandas as pd
import trafilatura

DB_PATH = "data/pages.db"
OUTPUT_CSV = "data/cleaned_pages.csv"

def extract_and_clean():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT url, title, text FROM pages WHERE http_status = 200", conn)
    conn.close()

    # Text is already extracted by trafilatura in run_crawl.py, just rename it
    df = df.rename(columns={"text": "cleaned_content"})
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Extracted readable text for {len(df)} pages → {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_and_clean()
