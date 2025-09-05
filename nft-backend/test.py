import requests
from datetime import datetime
from collections import defaultdict

url = 'https://api-mainnet.magiceden.dev/v3/rtp/monad-testnet/tokens/0x26c86f2835c114571df2b6ce9ba52296cc0fa6bb%3A0/activity/v5?limit=20&sortBy=eventTimestamp&sortDirection=asc&includeMetadata=true'
headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'accept': '*/*'
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print(f"Status: {response.status_code}")
    print(f"Activities found: {len(data.get('activities', []))}")
    print("-" * 80)
    
    # Filter and store relevant activities
    filtered_activities = []
    for activity in data.get('activities', []):
        activity_type = activity.get('type', 'Unknown')
        
        # Only keep mint, transfer, or sale activities
        if activity_type in ['mint', 'transfer', 'sale']:
            filtered_activities.append(activity)
    
    # Sort by timestamp first, then by log index for same transactions
    filtered_activities.sort(key=lambda x: (x.get('timestamp', 0), x.get('logIndex', 0)))
    
    # Track token holdings
    holdings = defaultdict(int)
    
    # Display the sorted activities and track holdings
    for activity in filtered_activities:
        activity_type = activity.get('type', 'Unknown')
        timestamp = activity.get('timestamp', 0)
        from_addr = activity.get('fromAddress')
        to_addr = activity.get('toAddress')
        amount = activity.get('amount', 1)
        
        # Update holdings based on activity type (always 1 token for NFTs)
        if activity_type == 'mint':
            # Mint adds 1 token to recipient
            if to_addr:
                holdings[to_addr] += 1
        elif activity_type in ['transfer', 'sale']:
            # Transfer/sale moves 1 token from sender to recipient
            if from_addr:
                holdings[from_addr] -= 1
            if to_addr:
                holdings[to_addr] += 1
        
        # Convert timestamp to readable date
        if timestamp:
            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        else:
            date_str = 'Unknown time'
        
        # Get price info
        price_info = activity.get('price', {})
        amount_info = price_info.get('amount', {})
        currency = price_info.get('currency', {})
        
        price_str = f"{amount_info.get('decimal', 0)} {currency.get('symbol', 'Unknown')}"
        
        # Get addresses for display
        from_display = from_addr[:10] + '...' if from_addr else 'N/A'
        to_display = to_addr[:10] + '...' if to_addr else 'N/A'
        
        log_index = activity.get('logIndex', 'N/A')
        tx_hash = activity.get('txHash', 'N/A')
        print(f"{activity_type.upper():12} | {date_str} | {price_str:15} | Log:{log_index:3} | {tx_hash} | {from_display} -> {to_display}")
    
    # Remove addresses with 0 tokens
    current_holders = {addr: count for addr, count in holdings.items() if count > 0}
    
    print("\n" + "=" * 80)
    print("CURRENT HOLDERS:")
    print("=" * 80)
    for addr, count in current_holders.items():
        print(f"{addr[:10]}... : {count} tokens")
    
    print(f"\nTotal unique holders: {len(current_holders)}")
        
else:
    print(f"Error {response.status_code}: {response.text}")