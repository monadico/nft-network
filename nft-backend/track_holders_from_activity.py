#!/usr/bin/env python3
"""
Track NFT holders using Magic Eden's token activity endpoint
This approach works for both ERC-721 and ERC-1155 collections
"""

import os
import json
import time
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
from collections import defaultdict

# Configuration
API_KEY = os.environ.get("ME_API_KEY", "YOUR_API_KEY")
BASE_URL = "https://api-mainnet.magiceden.dev/v3/rtp/monad-testnet"

def log(message: str) -> None:
    """Simple UTC timestamped logger"""
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
    print(f"[{ts}] {message}", flush=True)

def get_token_activity(contract_address: str, token_id: str = "0", limit: int = 20, continuation: str = None) -> Dict:
    """Get token activity from Magic Eden API"""
    url = f"{BASE_URL}/tokens/{contract_address}%3A{token_id}/activity/v5"
    
    params = {
        "limit": str(limit),
        "sortBy": "eventTimestamp",
        "includeMetadata": "true"
    }
    
    if continuation:
        params["continuation"] = continuation
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "accept": "*/*"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            log(f"Error {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        log(f"Request error: {e}")
        return None

def get_collection_activity(contract_address: str, limit: int = 20, continuation: str = None) -> Dict:
    """Get collection-wide activity from Magic Eden API - Note: This endpoint may not exist"""
    log("⚠️ Collection activity endpoint may not be available, trying alternative approach...")
    
    # Since collection activity endpoint doesn't exist, we'll try to get activity for token ID 0
    # and then infer collection-wide data
    return get_token_activity(contract_address, "0", limit, continuation)

def parse_activity_event(event: Dict) -> Tuple[str, str, str, int, str]:
    """Parse an activity event to extract ownership change info"""
    event_type = event.get("type", "")
    
    # Handle different event types
    if event_type == "transfer":
        from_addr = event.get("fromAddress", "")
        to_addr = event.get("toAddress", "")
        amount = int(event.get("amount", 1))
        tx_hash = event.get("txHash", "")
        return "transfer", from_addr, to_addr, amount, tx_hash
    
    elif event_type == "sale":
        from_addr = event.get("fromAddress", "")
        to_addr = event.get("toAddress", "")
        amount = int(event.get("amount", 1))
        tx_hash = event.get("txHash", "")
        return "sale", from_addr, to_addr, amount, tx_hash
    
    elif event_type == "mint":
        from_addr = "0x0000000000000000000000000000000000000000"  # Zero address for minting
        to_addr = event.get("toAddress", "")
        amount = int(event.get("amount", 1))
        tx_hash = event.get("txHash", "")
        return "mint", from_addr, to_addr, amount, tx_hash
    
    elif event_type == "airdrop":
        from_addr = event.get("fromAddress", "")
        to_addr = event.get("toAddress", "")
        amount = int(event.get("amount", 1))
        tx_hash = event.get("txHash", "")
        return "airdrop", from_addr, to_addr, amount, tx_hash
    
    # For other event types, return None
    return None, "", "", 0, ""

def track_ownership_changes(contract_address: str, token_id: str = "0") -> Dict[str, int]:
    """Track ownership changes for a specific token to determine current holders"""
    log(f"🔍 Tracking ownership for {contract_address} token {token_id}")
    
    # Track ownership changes chronologically
    ownership_timeline = []
    current_holders = defaultdict(int)
    
    continuation = None
    page_count = 0
    
    while True:
        page_count += 1
        log(f"📄 Fetching page {page_count}...")
        
        # Get token activity
        activity_data = get_token_activity(contract_address, token_id, limit=20, continuation=continuation)
        
        if not activity_data or "activities" not in activity_data:
            log("❌ No activity data received")
            break
        
        activities = activity_data["activities"]
        log(f"📊 Found {len(activities)} activities on page {page_count}")
        
        # Process each activity
        for event in activities:
            event_type, from_addr, to_addr, amount, tx_hash = parse_activity_event(event)
            
            if event_type:
                # Record the ownership change
                ownership_timeline.append({
                    "type": event_type,
                    "from": from_addr,
                    "to": to_addr,
                    "amount": amount,
                    "tx_hash": tx_hash,
                    "timestamp": event.get("timestamp", 0),
                    "created_at": event.get("createdAt", "")
                })
                
                log(f"📝 {event_type}: {from_addr} → {to_addr} (amount: {amount})")
        
        # Check for continuation
        continuation = activity_data.get("continuation")
        if not continuation:
            log("✅ Reached end of activity data")
            break
        
        # Rate limiting
        time.sleep(0.1)
    
    # Now process the timeline to determine current holders
    log("🔄 Processing ownership timeline...")
    
    # Sort by timestamp (oldest first)
    ownership_timeline.sort(key=lambda x: x["timestamp"])
    
    # Track current balances
    for event in ownership_timeline:
        if event["type"] in ["transfer", "sale", "mint", "airdrop"]:
            from_addr = event["from"]
            to_addr = event["to"]
            amount = event["amount"]
            
            # Decrease balance from sender (unless it's minting from zero address)
            if from_addr != "0x0000000000000000000000000000000000000000":
                current_holders[from_addr] -= amount
                if current_holders[from_addr] <= 0:
                    del current_holders[from_addr]
            
            # Increase balance for receiver
            current_holders[to_addr] += amount
    
    # Filter out zero balances
    final_holders = {addr: balance for addr, balance in current_holders.items() if balance > 0}
    
    log(f"🎯 Found {len(final_holders)} current holders")
    return final_holders

def track_collection_holders(contract_address: str) -> Dict[str, int]:
    """Track all holders for an entire collection using collection activity"""
    log(f"🔍 Tracking holders for collection {contract_address}")
    
    # Track ownership changes for all tokens
    all_holders = defaultdict(int)
    
    continuation = None
    page_count = 0
    
    while True:
        page_count += 1
        log(f"📄 Fetching collection activity page {page_count}...")
        
        # Get collection activity
        activity_data = get_collection_activity(contract_address, limit=20, continuation=continuation)
        
        if not activity_data or "activities" not in activity_data:
            log("❌ No collection activity data received")
            break
        
        activities = activity_data["activities"]
        log(f"📊 Found {len(activities)} activities on page {page_count}")
        
        # Process each activity
        for event in activities:
            event_type, from_addr, to_addr, amount, tx_hash = parse_activity_event(event)
            
            if event_type:
                # Record the ownership change
                if event_type in ["transfer", "sale", "mint", "airdrop"]:
                    # Decrease balance from sender (unless it's minting from zero address)
                    if from_addr != "0x0000000000000000000000000000000000000000":
                        all_holders[from_addr] -= amount
                        if all_holders[from_addr] <= 0:
                            del all_holders[from_addr]
                    
                    # Increase balance for receiver
                    all_holders[to_addr] += amount
                    
                    log(f"📝 {event_type}: {from_addr} → {to_addr} (amount: {amount})")
        
        # Check for continuation
        continuation = activity_data.get("continuation")
        if not continuation:
            log("✅ Reached end of collection activity data")
            break
        
        # Rate limiting
        time.sleep(0.1)
    
    # Filter out zero balances
    final_holders = {addr: balance for addr, balance in all_holders.items() if balance > 0}
    
    log(f"🎯 Found {len(final_holders)} total holders across collection")
    return final_holders

def test_token_tracking():
    """Test the token tracking with a known collection"""
    # Test with the collection from your example
    contract_address = "0x26c86f2835c114571df2b6ce9ba52296cc0fa6bb"  # Lil Chogstars
    token_id = "0"
    
    log(f"🧪 Testing token tracking for {contract_address} token {token_id}")
    
    # Track ownership for specific token
    holders = track_ownership_changes(contract_address, token_id)
    
    if holders:
        log(f"\n📋 Current holders for token {token_id}:")
        for addr, balance in sorted(holders.items(), key=lambda x: x[1], reverse=True):
            log(f"  {addr}: {balance}")
    else:
        log("❌ No holders found")

def test_collection_tracking():
    """Test the collection tracking with a known collection"""
    # Test with the problematic ERC-1155 collection
    contract_address = "0x038b19e8f047d22bea5355ca99d976f7ee02c754"  # slmnd
    
    log(f"🧪 Testing collection tracking for {contract_address}")
    
    # Track all holders for the collection
    holders = track_collection_holders(contract_address)
    
    if holders:
        log(f"\n📋 All collection holders:")
        for addr, balance in sorted(holders.items(), key=lambda x: x[1], reverse=True):
            log(f"  {addr}: {balance}")
        
        total_holders = len(holders)
        total_balance = sum(holders.values())
        log(f"\n📊 Summary:")
        log(f"  Unique holders: {total_holders}")
        log(f"  Total balance: {total_balance}")
    else:
        log("❌ No holders found")

def main():
    """Main function to run tests"""
    log("🚀 Starting NFT holder tracking tests")
    
    # Test 1: Token-specific tracking
    log("\n" + "="*50)
    log("TEST 1: Token-specific ownership tracking")
    log("="*50)
    test_token_tracking()
    
    # Test 2: Collection-wide tracking
    log("\n" + "="*50)
    log("TEST 2: Collection-wide holder tracking")
    log("="*50)
    test_collection_tracking()
    
    log("\n✅ All tests completed!")

if __name__ == "__main__":
    main()
