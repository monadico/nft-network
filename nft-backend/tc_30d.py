import requests
import json
from datetime import datetime

# Magic Eden API endpoint for trending collections on Monad testnet
BASE_URL = "https://api-mainnet.magiceden.dev/v3/rtp"
CHAIN = "monad-testnet"  # Correct chain identifier

def get_trending_collections(api_key=None, period="30d", limit=51, sort_by="volume"):
    """Fetch top selling collections from Magic Eden on Monad testnet"""
    
    url = f"{BASE_URL}/{CHAIN}/collections/trending/v1"
    
    # Query parameters
    params = {
        "period": period,
        "limit": limit,
        "sortBy": sort_by,
        "normalizeRoyalties": "false",
        "useNonFlaggedFloorAsk": "false"
    }
    
    headers = {
        "accept": "*/*",
        "User-Agent": "MagicEden-Trending-Client/1.0"
    }
    
    # Add API key if provided
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        print(f"🚀 Fetching trending collections from Magic Eden Monad Testnet")
        print(f"URL: {url}")
        print(f"Parameters: {params}")
        print("=" * 60)
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Debug: Print the raw response structure
            print(f"📊 Raw response type: {type(data)}")
            print(f"📊 Raw response: {json.dumps(data, indent=2)[:500]}...")
            print("=" * 60)
            
            # Handle different response structures
            if isinstance(data, list):
                collections = data
            elif isinstance(data, dict):
                # If it's a dict, look for collections in common keys
                collections = data.get('collections', data.get('data', data.get('result', [])))
                if not isinstance(collections, list):
                    collections = [data]  # If it's a single collection
            else:
                print(f"❌ Unexpected response format: {type(data)}")
                return None
            
            print(f"✅ Successfully retrieved {len(collections)} trending collections")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            # Display the collections
            for i, collection in enumerate(collections, 1):
                if isinstance(collection, dict):
                    print(f"\n{i}. Collection: {collection.get('name', collection.get('title', 'N/A'))}")
                    print(f"   Symbol: {collection.get('symbol', 'N/A')}")
                    print(f"   Floor Price: {collection.get('floorPrice', collection.get('floor', 'N/A'))}")
                    print(f"   Volume 24h: {collection.get('volume24h', collection.get('volume', 'N/A'))}")
                    print(f"   Listed Count: {collection.get('listedCount', collection.get('listed', 'N/A'))}")
                    
                    # Print additional details if available
                    if 'description' in collection:
                        desc = collection.get('description', 'N/A')
                        if desc and len(desc) > 100:
                            desc = desc[:100] + "..."
                        print(f"   Description: {desc}")
                    
                    # Print all available keys for debugging
                    print(f"   Available keys: {list(collection.keys())}")
                    print("-" * 40)
                else:
                    print(f"\n{i}. Collection: {collection} (type: {type(collection)})")
                    print("-" * 40)
            
            # Save to file for reference
            with open('trending_collections.json', 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n💾 Raw data saved to 'trending_collections.json'")
            
            return collections
            
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.RequestException as e:
        print(f"❌ Network error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_collection_stats(collection_symbol, api_key=None):
    """Get detailed stats for a specific collection"""
    
    url = f"{BASE_URL}/{CHAIN}/collections/{collection_symbol}/stats"
    
    headers = {
        "accept": "*/*",
        "User-Agent": "MagicEden-Trending-Client/1.0"
    }
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n📊 Detailed stats for {collection_symbol}:")
            print(json.dumps(data, indent=2))
            return data
        else:
            print(f"❌ Error getting stats for {collection_symbol}: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Magic Eden Monad Testnet Trending Collections")
    print("=" * 60)
    
    # You can add your API key here if you have one
    # API_KEY = "your_api_key_here"
    API_KEY = None  # Set to None if you don't have an API key
    
    # Get trending collections (limited to top 50 by volume)
    collections = get_trending_collections(api_key=API_KEY, limit=51)
    
    # Optional: Get detailed stats for a specific collection
    # Uncomment and modify the symbol below to get detailed stats
    # if collections:
    #     first_collection = collections[0].get('symbol') if collections else None
    #     if first_collection:
    #         get_collection_stats(first_collection, api_key=API_KEY)
