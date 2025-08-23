import os
import json
import time
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import contextlib

import requests

def log(message: str) -> None:
	# Simple UTC timestamped logger
	ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
	print(f"[{ts}] {message}", flush=True)

DB_PATH = Path('data/holders.sqlite')
BASE_URL = "https://api-mainnet.magiceden.dev/v3/rtp"
CHAIN = "monad-testnet"
API_KEY = os.environ.get("ME_API_KEY", "YOUR_API_KEY")
PAGE_LIMIT = 1000
RATE_PER_SEC = 2  # global requests per second

# Database schema from init_db.py
SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS collections (
	collection_id TEXT PRIMARY KEY,
	name TEXT,
	token_count INTEGER,
	owner_count INTEGER,
	processed_at TEXT
);

CREATE TABLE IF NOT EXISTS collection_holders (
	collection_id TEXT,
	holder_address TEXT,
	token_count INTEGER DEFAULT 1,
	PRIMARY KEY (collection_id, holder_address)
);

CREATE INDEX IF NOT EXISTS idx_collection_holders_collection ON collection_holders(collection_id);
CREATE INDEX IF NOT EXISTS idx_collection_holders_address ON collection_holders(holder_address);
"""

def ensure_database_schema():
	"""Ensure the database exists with the correct schema"""
	if not DB_PATH.exists():
		log("🔧 Database doesn't exist, creating with schema...")
		DB_PATH.parent.mkdir(parents=True, exist_ok=True)
		conn = sqlite3.connect(DB_PATH)
		conn.execute('PRAGMA foreign_keys = ON;')
		
		# Create schema
		for stmt in SCHEMA_SQL.strip().split(';'):
			stmt = stmt.strip()
			if stmt:
				conn.execute(stmt)
		conn.commit()
		conn.close()
		log("✅ Database schema created")
	else:
		# Check if tables exist
		conn = sqlite3.connect(DB_PATH)
		cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
		existing_tables = {row[0] for row in cursor.fetchall()}
		required_tables = {'collections', 'collection_holders'}
		
		if not required_tables.issubset(existing_tables):
			log("🔧 Database exists but missing required tables, creating schema...")
			conn.execute('PRAGMA foreign_keys = ON;')
			
			# Create missing tables
			for stmt in SCHEMA_SQL.strip().split(';'):
				stmt = stmt.strip()
				if stmt:
					conn.execute(stmt)
			conn.commit()
			log("✅ Database schema updated")
		
		conn.close()


class RateLimiter:
	"""Simple rate limiter for requests."""
	def __init__(self, rate_per_sec: int):
		self._rate = rate_per_sec
		self._last_request = 0.0

	def acquire(self) -> None:
		"""Wait if necessary to respect rate limit."""
		now = time.time()
		time_since_last = now - self._last_request
		min_interval = 1.0 / self._rate
		
		if time_since_last < min_interval:
			sleep_time = min_interval - time_since_last
			time.sleep(sleep_time)
		
		self._last_request = time.time()


def fetch_tokens_page(limiter: RateLimiter, collection_address: str, offset: int) -> Dict:
	limiter.acquire()
	url = f"{BASE_URL}/{CHAIN}/tokens/v6"
	params = {
		"collection": collection_address,
		"sortBy": "updatedAt",
		"sortDirection": "asc",
		"limit": str(PAGE_LIMIT),
		"offset": str(offset),
		"includeTopBid": "false",
		"excludeEOA": "false",
		"includeAttributes": "false",
		"includeQuantity": "false",
		"includeDynamicPricing": "false",
		"includeLastSale": "false",
		"normalizeRoyalties": "false",
	}
	headers = {
		"accept": "*/*",
		"Authorization": f"Bearer {API_KEY}",
	}
	for attempt in range(5):
		try:
			resp = requests.get(url, headers=headers, params=params, timeout=60)
			if resp.status_code == 200:
				return resp.json()
			elif resp.status_code == 429:
				log(f"{collection_address} offset={offset}: 429 rate limited. Backing off 5s (attempt {attempt+1}/5)")
				time.sleep(5)
				continue
			elif 500 <= resp.status_code < 600:
				_backoff = min(2 ** attempt, 10)
				log(f"{collection_address} offset={offset}: {resp.status_code} server error. Backing off {_backoff}s (attempt {attempt+1}/5)")
				time.sleep(_backoff)
				continue
			else:
				text = resp.text
				raise RuntimeError(f"HTTP {resp.status_code}: {text[:200]}")
		except (requests.RequestException, requests.Timeout) as e:
			_backoff = min(2 ** attempt, 10)
			log(f"{collection_address} offset={offset}: network error '{e}'. Backing off {_backoff}s (attempt {attempt+1}/5)")
			time.sleep(_backoff)
			continue
	raise RuntimeError(f"Failed to fetch {collection_address} offset={offset} after 5 attempts")


MAX_PAGES_PER_COLLECTION = 300


def upsert_collection_holders(conn: sqlite3.Connection, collection_id: str, holder_counts: Dict[str, int], total_tokens: int) -> None:
	"""Upsert holders and collection_holders for a collection as a single transaction."""
	processed_at = datetime.now(timezone.utc).isoformat()
	with conn:
		# Replace current snapshot for this collection
		conn.execute("DELETE FROM collection_holders WHERE collection_id = ?", (collection_id,))
		
		# Insert holder data directly
		if holder_counts:
			conn.executemany(
				"INSERT INTO collection_holders(collection_id, holder_address, token_count) VALUES(?, ?, ?)",
				[(collection_id, addr, count) for addr, count in holder_counts.items()],
			)
		
		# Update collections summary
		conn.execute(
			"UPDATE collections SET token_count = ?, owner_count = ?, processed_at = ? WHERE collection_id = ?",
			(total_tokens, len(holder_counts), processed_at, collection_id),
		)


def parse_owner_from_item(item: Dict) -> str | None:
	# Expected structure: { "token": { ..., "owner": "0x..." } }
	t = item.get("token") if isinstance(item, dict) else None
	if isinstance(t, dict):
		owner = t.get("owner") or t.get("ownerAddress") or t.get("currentOwner")
		if isinstance(owner, str):
			return owner
	# Fallbacks if structure differs
	for key in ("owner", "ownerAddress", "currentOwner"):
		val = item.get(key) if isinstance(item, dict) else None
		if isinstance(val, str):
			return val
	return None


def process_collection(limiter: RateLimiter, conn_path: Path, collection_id: str, name: str | None, token_count_hint: int | None) -> Tuple[str, int, int]:
	holder_counts: Dict[str, int] = {}
	total = 0
	offset = 0
	pages = 0
	log(f"▶ Start collection {collection_id}{f' ({name})' if name else ''}")
	while True:
		data = fetch_tokens_page(limiter, collection_id, offset)
		tokens = []
		if isinstance(data, dict):
			tokens = data.get("tokens", [])
		elif isinstance(data, list):
			tokens = data
		log(f"{collection_id}: page offset={offset} fetched {len(tokens)} tokens")
		if not tokens:
			break
		for item in tokens:
			owner = parse_owner_from_item(item)
			if owner:
				holder_counts[owner] = holder_counts.get(owner, 0) + 1
			total += 1
		pages += 1

		# Stop if we've seen as many tokens as expected
		if token_count_hint is not None and total >= token_count_hint:
			log(f"{collection_id}: reached token_count hint {token_count_hint}, stopping pagination")
			break
		
		if len(tokens) < PAGE_LIMIT:
			break
		# Safety cap
		if pages >= MAX_PAGES_PER_COLLECTION:
			log(f"{collection_id}: reached safety cap of {MAX_PAGES_PER_COLLECTION} pages, stopping early")
			break
		offset += PAGE_LIMIT
	# Write to DB directly since we're not async anymore
	conn = sqlite3.connect(conn_path)
	conn.execute('PRAGMA foreign_keys = ON;')
	upsert_collection_holders(conn, collection_id, holder_counts, total)
	conn.close()
	log(f"✔ Saved to DB {collection_id}: tokens={total}, unique_holders={len(holder_counts)}")
	return collection_id, total, len(holder_counts)


def main():
	start = time.time()
	
	# Ensure database schema exists
	ensure_database_schema()
	
	# Read collections from trending_collections.json (all collections)
	with open("trending_collections.json", "r") as f:
		trending = json.load(f)
	# Remove Nad Name Service by id (case-insensitive)
	excluded_id = "0x3019bf1dfb84e5b46ca9d0eec37de08a59a41308"
	collections_data = [c for c in trending["collections"] if str(c.get("id", "")).lower() != excluded_id]
	collections: List[Tuple[str, str | None, int | None]] = [
		(c["id"], c.get("name"), c.get("tokenCount")) for c in collections_data
	]
	log(f"Found {len(collections)} collections to process (rate={RATE_PER_SEC}/s)")
	limiter = RateLimiter(RATE_PER_SEC)
	results: List[Tuple[str, int, int]] = []
	_progress = {"done": 0, "total": len(collections)}
	
	for cid, name, tok in collections:
		try:
			cid_, total, holders = process_collection(limiter, DB_PATH, cid, name, tok)
			log(f"✅ Done {cid_}: tokens={total}, holders={holders}")
			results.append((cid_, total, holders))
		except Exception as e:
			log(f"❌ {cid}: {e}")
		finally:
			_progress["done"] += 1
			if _progress["done"] % 10 == 0 or _progress["done"] == _progress["total"]:
				log(f"Progress: {_progress['done']}/{_progress['total']} collections processed")
	
	log("Done processing all collections")
	log(f"Total time: {time.time() - start:.1f}s")

if __name__ == "__main__":
	main()
