import sqlite3
import json
from pathlib import Path

def test_database_results():
    """Test if the fetch_holders.py script worked properly"""
    
    db_path = Path('data/holders.sqlite')
    
    if not db_path.exists():
        print("❌ Database file not found!")
        return
    
    print("🔍 Testing database results...")
    print("=" * 50)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"📋 Tables found: {[table[0] for table in tables]}")
    print()
    
    # Check collections table
    cursor.execute("SELECT collection_id, name, token_count, owner_count, processed_at FROM collections")
    collections = cursor.fetchall()
    
    print("📊 Collections processed:")
    for collection in collections:
        collection_id, name, token_count, owner_count, processed_at = collection
        print(f"  • {name or 'Unknown'}")
        print(f"    ID: {collection_id}")
        print(f"    Tokens: {token_count}")
        print(f"    Unique Holders: {owner_count}")
        print(f"    Processed: {processed_at}")
        print()
    
    # Check holders table
    cursor.execute("SELECT COUNT(*) FROM holders")
    total_holders = cursor.fetchone()[0]
    print(f"👥 Total unique holders across all collections: {total_holders}")
    
    # Check collection_holders table
    cursor.execute("""
        SELECT c.name, COUNT(ch.holder_id) as holder_count, SUM(ch.token_count) as total_tokens
        FROM collection_holders ch
        JOIN collections c ON ch.collection_id = c.collection_id
        GROUP BY ch.collection_id
    """)
    collection_stats = cursor.fetchall()
    
    print("\n📈 Collection Statistics:")
    for name, holder_count, total_tokens in collection_stats:
        print(f"  • {name or 'Unknown'}: {holder_count} holders, {total_tokens} tokens")
    
    # Show some sample holders
    cursor.execute("""
        SELECT h.address, ch.token_count, c.name
        FROM collection_holders ch
        JOIN holders h ON ch.holder_id = h.holder_id
        JOIN collections c ON ch.collection_id = c.collection_id
        ORDER BY ch.token_count DESC
        LIMIT 5
    """)
    top_holders = cursor.fetchall()
    
    print("\n🏆 Top 5 holders by token count:")
    for address, token_count, collection_name in top_holders:
        print(f"  • {address[:10]}...{address[-8:]}: {token_count} tokens in {collection_name}")
    
    conn.close()
    
    # Also check the trending collections file
    print("\n" + "=" * 50)
    print("📄 Checking trending_collections.json...")
    
    try:
        with open('trending_collections.json', 'r') as f:
            trending_data = json.load(f)
        
        if 'collections' in trending_data:
            collections_data = trending_data['collections'][:2]  # First 2 only
            print(f"✅ Found {len(collections_data)} collections in trending data:")
            for i, collection in enumerate(collections_data, 1):
                print(f"  {i}. {collection.get('name', 'Unknown')} ({collection.get('symbol', 'N/A')})")
                print(f"     ID: {collection.get('id', 'N/A')}")
                print(f"     Token Count: {collection.get('tokenCount', 'N/A')}")
        else:
            print("⚠️  No 'collections' key found in trending data")
            
    except Exception as e:
        print(f"❌ Error reading trending_collections.json: {e}")

if __name__ == "__main__":
    test_database_results()
