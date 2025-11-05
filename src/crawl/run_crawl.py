# src/crawl/run_crawl.py
import hashlib
import time
import csv
from pathlib import Path

import scrapy
from scrapy.crawler import CrawlerProcess
from trafilatura import extract as trafi_extract

# Reuse your SQLite helper
from src.storage.sqlite_db import get_connection

CSV_PATH = Path("data/incorta_links_master.csv")  # already created earlier

def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()

class CsvSeedSpider(scrapy.Spider):
    name = "csv_seed"
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_TIMEOUT": 20,
        "CONCURRENT_REQUESTS": 8,
        "AUTOTHROTTLE_ENABLED": True,
        "DEFAULT_REQUEST_HEADERS": {
            "User-Agent": "IbnBattouta/0.1 (+research; contact=incorta-search)"
        },
        "LOG_LEVEL": "INFO",
    }

    def start_requests(self):
        assert CSV_PATH.exists(), f"CSV not found: {CSV_PATH}"
        with CSV_PATH.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row["url"].strip()
                # Add https:// if missing
                if url and not url.startswith(("http://", "https://")):
                    url = "https://" + url
                title = (row.get("title") or "").strip()
                source = (row.get("source") or "").strip()
                yield scrapy.Request(
                    url=url,
                    callback=self.parse_page,
                    cb_kwargs={"title": title, "source": source},
                    errback=self.errback_log,
                    dont_filter=True,
                )

    def errback_log(self, failure):
        req = failure.request
        self.logger.warning(f"Fetch failed {req.url}: {repr(failure.value)}")
        self._upsert_page(url=req.url, source=req.cb_kwargs.get("source",""),
                          title=req.cb_kwargs.get("title",""),
                          text="", http_status=-1)

    def parse_page(self, response, title, source):
        http_status = response.status
        html = response.text
        # Clean main content with trafilatura
        text = trafi_extract(html, include_comments=False, include_tables=False) or ""
        text = text.strip()
        self._upsert_page(url=response.url, source=source, title=title,
                          text=text, http_status=http_status)

    def _upsert_page(self, url, source, title, text, http_status):
        con = get_connection()
        cur = con.cursor()
        digest = sha256_text(text)
        fetch_ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        # Insert or update if changed
        cur.execute("SELECT sha256 FROM pages WHERE url = ?", (url,))
        row = cur.fetchone()
        if row is None:
            cur.execute(
                """INSERT INTO pages (url, source, title, text, sha256, http_status, fetch_ts, indexed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
                (url, source, title, text, digest, http_status, fetch_ts),
            )
        else:
            if row["sha256"] != digest or http_status != 200:
                cur.execute(
                    """UPDATE pages
                       SET source=?, title=?, text=?, sha256=?, http_status=?, fetch_ts=?, indexed=0
                       WHERE url=?""",
                    (source, title, text, digest, http_status, fetch_ts, url),
                )
        # Keep the small FTS index on titles for hybrid keyword recall
        cur.execute("INSERT INTO fts_pages (title, url, source) VALUES (?, ?, ?)", (title, url, source))
        con.commit()
        con.close()

if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(CsvSeedSpider)
    process.start()
