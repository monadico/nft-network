import sqlite3
import sys
from pathlib import Path

DB_PATH = Path('data/holders.sqlite')

def get_collection_info(conn, collection_id=None, name=None):
    if collection_id:
        row = conn.execute("SELECT collection_id, name, token_count, owner_count, processed_at FROM collections WHERE collection_id = ?", (collection_id,)).fetchone()
    elif name:
        row = conn.execute("SELECT collection_id, name, token_count, owner_count, processed_at FROM collections WHERE name = ?", (name,)).fetchone()
    else:
        raise ValueError("Must provide collection_id or name")
    if not row:
        print("Collection not found.")
        return
    print(f"Collection ID: {row[0]}")
    print(f"Name: {row[1]}")
    print(f"Token count: {row[2]}")
    print(f"Owner count: {row[3]}")
    print(f"Last processed: {row[4]}")
    print("\nTop holders:")
    holders = conn.execute('''
        SELECT h.address, ch.token_count
        FROM collection_holders ch
        JOIN holders h ON ch.holder_id = h.holder_id
        WHERE ch.collection_id = ?
        ORDER BY ch.token_count DESC, h.address ASC
        LIMIT 50
    ''', (row[0],)).fetchall()
    for addr, count in holders:
        print(f"  {addr}: {count}")
    if not holders:
        print("  (No holders found)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Query collection details from holders DB.")
    parser.add_argument('--id', help='Collection ID')
    parser.add_argument('--name', help='Collection name')
    args = parser.parse_args()
    if not args.id and not args.name:
        print("Please provide --id or --name.")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    try:
        get_collection_info(conn, collection_id=args.id, name=args.name)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
