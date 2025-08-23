import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path('data/holders.sqlite')
TRENDING_PATH = Path('trending_collections.json')

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

def get_conn():
	conn = sqlite3.connect(DB_PATH)
	conn.execute('PRAGMA foreign_keys = ON;')
	return conn

def init_schema(conn: sqlite3.Connection) -> None:
	for stmt in SCHEMA_SQL.strip().split(';'):
		stmt = stmt.strip()
		if stmt:
			conn.execute(stmt)
	conn.commit()


def ingest_trending(conn: sqlite3.Connection, trending_path: Path) -> int:
	if not trending_path.exists():
		raise FileNotFoundError(f"Missing {trending_path}")

	with trending_path.open('r') as f:
		data = json.load(f)

	collections = data.get('collections', [])
	now_iso = datetime.utcnow().isoformat()
	inserted = 0

	with conn:
		for c in collections:
			collection_id = c.get('id')
			name = c.get('name')
			token_count = c.get('tokenCount')
			owner_count = c.get('ownerCount')
			if not collection_id:
				continue
			conn.execute(
				"""
				INSERT INTO collections(collection_id, name, token_count, owner_count, processed_at)
				VALUES(?, ?, ?, ?, ?)
				ON CONFLICT(collection_id) DO UPDATE SET
					name=excluded.name,
					token_count=excluded.token_count,
					owner_count=excluded.owner_count,
					processed_at=excluded.processed_at
				""",
				(collection_id, name, token_count, owner_count, now_iso),
			)
			inserted += 1
	return inserted


def main():
	DB_PATH.parent.mkdir(parents=True, exist_ok=True)
	conn = get_conn()
	try:
		init_schema(conn)
		num = ingest_trending(conn, TRENDING_PATH)
		print(f"Ingested {num} collections into {DB_PATH}")
	finally:
		conn.close()

if __name__ == '__main__':
	main()
