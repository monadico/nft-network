#!/usr/bin/env python3
"""
ERC-1155 Holder Retrieval using Blockchain RPC calls
Bypasses Magic Eden API limitations for ERC-1155 collections
"""

import os
import json
import time
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests

# ERC-1155 ABI for balanceOf function
ERC1155_ABI = [
    {
        "constant": True,
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "id", "type": "uint256"}
        ],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "id", "type": "uint256"}],
        "name": "totalSupply",
        "outputs": [{"name": "", "type": "uint256"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
]

# Monad RPC endpoint
MONAD_RPC = "https://rpc.testnet.monad.xyz"

def log(message: str) -> None:
    """Simple UTC timestamped logger"""
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
    print(f"[{ts}] {message}", flush=True)

def get_contract_balance(contract_address: str, token_id: str = "0") -> int:
    """Get total supply of an ERC-1155 token"""
    try:
        # Encode the function call for totalSupply(id)
        function_selector = "0xbd85b039"  # totalSupply(uint256)
        token_id_hex = hex(int(token_id))[2:].zfill(64)
        
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [
                {
                    "to": contract_address,
                    "data": function_selector + token_id_hex
                },
                "latest"
            ],
            "id": 1
        }
        
        response = requests.post(MONAD_RPC, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if "result" in result and result["result"] != "0x":
                # Convert hex to decimal
                balance = int(result["result"], 16)
                return balance
        return 0
    except Exception as e:
        log(f"Error getting contract balance: {e}")
        return 0

def get_holder_balance(contract_address: str, holder_address: str, token_id: str = "0") -> int:
    """Get balance of a specific holder for an ERC-1155 token"""
    try:
        # Encode the function call for balanceOf(account, id)
        function_selector = "0x00fdd58e"  # balanceOf(address,uint256)
        holder_hex = holder_address[2:].zfill(64)  # Remove 0x and pad
        token_id_hex = hex(int(token_id))[2:].zfill(64)
        
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [
                {
                    "to": contract_address,
                    "data": function_selector + holder_hex + token_id_hex
                },
                "latest"
            ],
            "id": 1
        }
        
        response = requests.post(MONAD_RPC, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if "result" in result and result["result"] != "0x":
                # Convert hex to decimal
                balance = int(result["result"], 16)
                return balance
        return 0
    except Exception as e:
        log(f"Error getting holder balance for {holder_address}: {e}")
        return 0

def get_transfer_events(contract_address: str, from_block: int = 0, to_block: str = "latest") -> List[Dict]:
    """Get TransferSingle and TransferBatch events to find holders"""
    try:
        # TransferSingle event signature
        transfer_single_topic = "0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62"
        
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getLogs",
            "params": [
                {
                    "address": contract_address,
                    "topics": [transfer_single_topic],
                    "fromBlock": hex(from_block),
                    "toBlock": to_block
                }
            ],
            "id": 1
        }
        
        response = requests.post(MONAD_RPC, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if "result" in result:
                return result["result"]
        return []
    except Exception as e:
        log(f"Error getting transfer events: {e}")
        return []

def parse_transfer_event(event: Dict) -> Tuple[str, str, int]:
    """Parse TransferSingle event to extract from, to, and amount"""
    try:
        # TransferSingle event: TransferSingle(operator, from, to, id, value)
        # Topics: [event_signature, operator, from, to]
        # Data: [id, value]
        
        if len(event["topics"]) >= 4:
            from_addr = "0x" + event["topics"][2][-40:]  # from address
            to_addr = "0x" + event["topics"][3][-40:]     # to address
            
            # Parse data for token ID and value
            data = event["data"]
            if len(data) >= 66:  # 32 bytes for id + 32 bytes for value
                token_id = int(data[:66], 16)
                value = int(data[66:], 16)
                return from_addr, to_addr, value
        
        return "", "", 0
    except Exception as e:
        log(f"Error parsing transfer event: {e}")
        return "", "", 0

def get_erc1155_holders(contract_address: str, token_id: str = "0") -> Dict[str, int]:
    """Get all holders of an ERC-1155 token using blockchain data"""
    log(f"🔍 Getting ERC-1155 holders for {contract_address} token {token_id}")
    
    holder_balances: Dict[str, int] = {}
    
    try:
        # Method 1: Get transfer events to find addresses that ever held the token
        log("📡 Getting transfer events...")
        events = get_transfer_events(contract_address)
        log(f"Found {len(events)} transfer events")
        
        # Track all addresses that appeared in transfers
        addresses_to_check = set()
        
        for event in events:
            from_addr, to_addr, value = parse_transfer_event(event)
            if from_addr and to_addr:
                addresses_to_check.add(from_addr)
                addresses_to_check.add(to_addr)
        
        log(f"Found {len(addresses_to_check)} unique addresses in transfer events")
        
        # Method 2: Check current balances for all addresses that ever held the token
        log("🔍 Checking current balances...")
        for addr in addresses_to_check:
            if addr != "0x0000000000000000000000000000000000000000":  # Skip zero address
                balance = get_holder_balance(contract_address, addr, token_id)
                if balance > 0:
                    holder_balances[addr] = balance
                    log(f"✅ {addr}: {balance}")
        
        # Method 3: Also check the contract creator and any known addresses
        log("🔍 Checking additional addresses...")
        additional_addresses = [
            "0xed204e46fb54d3cadb7ee5d2155862ea89903a36",  # Creator from API
            "0x0053ad828413aca31fad0d4087fa225368b72e6c",  # Owner from API
        ]
        
        for addr in additional_addresses:
            if addr not in holder_balances:
                balance = get_holder_balance(contract_address, addr, token_id)
                if balance > 0:
                    holder_balances[addr] = balance
                    log(f"✅ {addr}: {balance}")
        
        log(f"🎯 Total holders found: {len(holder_balances)}")
        return holder_balances
        
    except Exception as e:
        log(f"❌ Error getting ERC-1155 holders: {e}")
        return {}

def test_erc1155_holders():
    """Test the ERC-1155 holder retrieval with the problematic collection"""
    contract_address = "0x038b19e8f047d22bea5355ca99d976f7ee02c754"  # slmnd
    
    log(f"🧪 Testing ERC-1155 holder retrieval for {contract_address}")
    
    # Get total supply first
    total_supply = get_contract_balance(contract_address)
    log(f"📊 Total supply: {total_supply}")
    
    # Get holders
    holders = get_erc1155_holders(contract_address)
    
    log(f"\n📋 Holder Summary:")
    log(f"Total supply: {total_supply}")
    log(f"Unique holders: {len(holders)}")
    log(f"Total distributed: {sum(holders.values())}")
    
    if holders:
        log(f"\n👥 Top holders:")
        sorted_holders = sorted(holders.items(), key=lambda x: x[1], reverse=True)
        for addr, balance in sorted_holders[:10]:
            log(f"  {addr}: {balance}")
    
    return holders

if __name__ == "__main__":
    test_erc1155_holders()
