#!/usr/bin/env python3
"""
NFT Collection Cluster Analysis using Graph Theory

This script analyzes relationships between NFT collections based on shared holders.
It represents collections as nodes in a graph and shared holders as weighted edges.

Main Analysis Method: DEGREE CENTRALITY
- Measures the number of connections (weighted or unweighted) a collection has
- High degree = influential in terms of direct shared holders
- Simple and direct approach suitable for 50+ collections
"""

import sqlite3
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, Set, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
DB_PATH = Path('data/holders.sqlite')
OUTPUT_DIR = Path('analysis_results')
SIMILARITY_THRESHOLD = 0.01  # Minimum Jaccard similarity to create an edge

def setup_output_directory():
    """Create output directory for analysis results"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"=Output directory: {OUTPUT_DIR}")

def load_collection_data() -> pd.DataFrame:
    """
    Load collection metadata from the database
    
    Returns:
        DataFrame with collection_id, name, and owner_count
    """
    print("=� Loading collection data from database...")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get collection metadata
    collections_df = pd.read_sql_query("""
        SELECT collection_id, name, owner_count, processed_at
        FROM collections
        WHERE owner_count > 0
        ORDER BY owner_count DESC
    """, conn)
    
    conn.close()
    
    print(f" Loaded {len(collections_df)} collections")
    print(f"   Total holders range: {collections_df['owner_count'].min()} - {collections_df['owner_count'].max()}")
    
    return collections_df

def load_holder_data() -> Dict[str, Set[str]]:
    """
    Load holder data and organize by collection
    
    Returns:
        Dictionary mapping collection_id -> set of holder addresses
    """
    print("=e Loading holder data from database...")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get all holder relationships
    holders_df = pd.read_sql_query("""
        SELECT collection_id, holder_address
        FROM collection_holders
    """, conn)
    
    conn.close()
    
    # Group holders by collection
    collection_holders = {}
    for _, row in holders_df.iterrows():
        collection_id = row['collection_id']
        holder_address = row['holder_address']
        
        if collection_id not in collection_holders:
            collection_holders[collection_id] = set()
        
        collection_holders[collection_id].add(holder_address)
    
    print(f" Loaded holder data for {len(collection_holders)} collections")
    print(f"   Total unique holder relationships: {len(holders_df)}")
    
    return collection_holders

def calculate_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Calculate Jaccard similarity coefficient between two sets
    
    Jaccard Index = |A ) B| / |A * B|
    - Measures similarity between finite sample sets
    - Range: 0 (no overlap) to 1 (identical sets)
    
    Args:
        set_a: First set of holders
        set_b: Second set of holders
        
    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def build_similarity_graph(collections_df: pd.DataFrame, collection_holders: Dict[str, Set[str]]) -> nx.Graph:
    """
    Build a weighted graph representing collection relationships
    
    GRAPH STRUCTURE:
    - Nodes: NFT collections (collection_id)
    - Edges: Shared holder relationships
    - Weights: Jaccard similarity coefficient
    
    Args:
        collections_df: Collection metadata
        collection_holders: Holder sets per collection
        
    Returns:
        NetworkX Graph with weighted edges
    """
    print("< Building collection similarity graph...")
    print(f"   Method: Jaccard similarity (threshold: {SIMILARITY_THRESHOLD})")
    
    # Initialize empty graph
    G = nx.Graph()
    
    # Add nodes (collections) with attributes
    for _, collection in collections_df.iterrows():
        collection_id = collection['collection_id']
        G.add_node(collection_id, 
                  name=collection['name'], 
                  holder_count=collection['owner_count'])
    
    # Calculate pairwise similarities and add edges
    collections = list(collections_df['collection_id'])
    edges_added = 0
    total_comparisons = len(collections) * (len(collections) - 1) // 2
    
    print(f"   Computing {total_comparisons} pairwise similarities...")
    
    for i, collection_a in enumerate(collections):
        for j, collection_b in enumerate(collections[i+1:], i+1):
            # Get holder sets
            holders_a = collection_holders.get(collection_a, set())
            holders_b = collection_holders.get(collection_b, set())
            
            # Skip if either collection has no holders
            if not holders_a or not holders_b:
                continue
            
            # Calculate similarity
            similarity = calculate_jaccard_similarity(holders_a, holders_b)
            
            # Add edge if similarity exceeds threshold
            if similarity >= SIMILARITY_THRESHOLD:
                G.add_edge(collection_a, collection_b, 
                          weight=similarity,
                          shared_holders=len(holders_a.intersection(holders_b)))
                edges_added += 1
    
    print(f" Graph construction complete:")
    print(f"   Nodes (collections): {G.number_of_nodes()}")
    print(f"   Edges (relationships): {G.number_of_edges()}")
    print(f"   Edge density: {nx.density(G):.4f}")
    
    return G

def analyze_degree_centrality(G: nx.Graph, collections_df: pd.DataFrame) -> pd.DataFrame:
    """
    DEGREE CENTRALITY ANALYSIS
    
    Degree centrality measures the importance of a node based on the number of connections.
    
    MATHEMATICAL DEFINITION:
    - Unweighted: C_D(v) = deg(v) / (n-1)
    - Weighted: C_D(v) = � w(u,v) for all neighbors u
    
    INTERPRETATION:
    - Higher values indicate collections with more shared holder relationships
    - Collections with high degree centrality are "hubs" in the holder ecosystem
    - Useful for identifying influential or popular collections
    
    Args:
        G: NetworkX graph
        collections_df: Collection metadata
        
    Returns:
        DataFrame with degree centrality scores
    """
    print("=� Computing Degree Centrality...")
    print("   Method: Measures direct connections to other collections")
    print("   Interpretation: Higher scores = more shared holder relationships")
    
    # Calculate unweighted degree centrality
    unweighted_centrality = nx.degree_centrality(G)
    
    # Calculate weighted degree (sum of edge weights)
    weighted_degree = {}
    for node in G.nodes():
        weighted_degree[node] = sum(G[node][neighbor]['weight'] 
                                  for neighbor in G.neighbors(node))
    
    # Normalize weighted degree by max possible connections
    max_weighted_degree = max(weighted_degree.values()) if weighted_degree.values() else 1
    normalized_weighted_degree = {node: score / max_weighted_degree 
                                for node, score in weighted_degree.items()}
    
    # Create results DataFrame
    results = []
    for _, collection in collections_df.iterrows():
        collection_id = collection['collection_id']
        
        results.append({
            'collection_id': collection_id,
            'collection_name': collection['name'],
            'holder_count': collection['owner_count'],
            'degree_centrality_unweighted': unweighted_centrality.get(collection_id, 0),
            'degree_centrality_weighted': normalized_weighted_degree.get(collection_id, 0),
            'total_connections': len(list(G.neighbors(collection_id))) if collection_id in G else 0,
            'avg_edge_weight': np.mean([G[collection_id][neighbor]['weight'] 
                                       for neighbor in G.neighbors(collection_id)]) 
                             if collection_id in G and len(list(G.neighbors(collection_id))) > 0 else 0
        })
    
    results_df = pd.DataFrame(results)
    
    # Sort by weighted degree centrality (primary metric)
    results_df = results_df.sort_values('degree_centrality_weighted', ascending=False)
    
    print(f" Degree centrality analysis complete")
    print(f"   Top collection by weighted degree: {results_df.iloc[0]['collection_name']}")
    print(f"   Max connections: {results_df['total_connections'].max()}")
    
    return results_df

# ==================================================================================
# EIGENVECTOR CENTRALITY ANALYSIS
# ==================================================================================

def analyze_eigenvector_centrality(G: nx.Graph, collections_df: pd.DataFrame) -> pd.DataFrame:
    """
    EIGENVECTOR CENTRALITY ANALYSIS
    
    Eigenvector centrality measures the importance of a node based on connections to other 
    important nodes (like PageRank for graphs).
    
    MATHEMATICAL DEFINITION:
    - A node's score is proportional to the sum of its neighbors' scores
    - Formula: Ax = lambda * x (where A is adjacency matrix, x is centrality vector)
    - Iteratively computed until convergence
    
    INTERPRETATION:
    - Higher values indicate collections connected to other influential collections
    - Captures "indirect influence" in the holder ecosystem
    - A collection sharing holders with many high-overlap collections gets higher score
    - More sophisticated than degree centrality as it considers neighbor importance
    
    EXAMPLE:
    - Collection A connected to 5 minor collections
    - Collection B connected to 3 major hub collections
    - Eigenvector centrality: B > A (despite A having more connections)
    
    Args:
        G: NetworkX graph with weighted edges
        collections_df: Collection metadata
        
    Returns:
        DataFrame with eigenvector centrality scores
    """
    print("Computing Eigenvector Centrality...")
    print("   Method: Measures connections to other influential collections")
    print("   Interpretation: Higher scores = connected to important holder hubs")
    
    try:
        # Calculate unweighted eigenvector centrality
        unweighted_eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        
        # Calculate weighted eigenvector centrality
        weighted_eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        
    except nx.NetworkXError as e:
        print(f"   Warning: Eigenvector centrality computation failed: {e}")
        print("   This can happen with disconnected graphs or numerical issues")
        # Fallback to zeros
        unweighted_eigenvector = {node: 0.0 for node in G.nodes()}
        weighted_eigenvector = {node: 0.0 for node in G.nodes()}
    
    # Create results DataFrame
    results = []
    for _, collection in collections_df.iterrows():
        collection_id = collection['collection_id']
        
        # Calculate neighbor influence score (sum of neighbor eigenvector scores)
        neighbor_influence = 0
        if collection_id in G:
            for neighbor in G.neighbors(collection_id):
                neighbor_influence += weighted_eigenvector.get(neighbor, 0)
        
        results.append({
            'collection_id': collection_id,
            'collection_name': collection['name'],
            'holder_count': collection['owner_count'],
            'eigenvector_centrality_unweighted': unweighted_eigenvector.get(collection_id, 0),
            'eigenvector_centrality_weighted': weighted_eigenvector.get(collection_id, 0),
            'total_connections': len(list(G.neighbors(collection_id))) if collection_id in G else 0,
            'neighbor_influence_score': neighbor_influence,
            'avg_neighbor_centrality': neighbor_influence / len(list(G.neighbors(collection_id))) 
                                     if collection_id in G and len(list(G.neighbors(collection_id))) > 0 else 0
        })
    
    results_df = pd.DataFrame(results)
    
    # Sort by weighted eigenvector centrality (primary metric)
    results_df = results_df.sort_values('eigenvector_centrality_weighted', ascending=False)
    
    print(f"Eigenvector centrality analysis complete")
    if len(results_df) > 0:
        print(f"   Top collection by weighted eigenvector: {results_df.iloc[0]['collection_name']}")
        print(f"   Max eigenvector score: {results_df['eigenvector_centrality_weighted'].max():.4f}")
    
    return results_df

# ==================================================================================
# COMBINED ANALYSIS AND RESULTS
# ==================================================================================

def save_results(degree_results_df: pd.DataFrame, eigenvector_results_df: pd.DataFrame, G: nx.Graph):
    """Save analysis results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save degree centrality results
    csv_path = OUTPUT_DIR / f"degree_centrality_analysis_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"=� Results saved to: {csv_path}")
    
    # Save top 10 summary
    top_10 = results_df.head(10)
    print("\n<� TOP 10 COLLECTIONS BY DEGREE CENTRALITY:")
    print("=" * 80)
    for i, row in top_10.iterrows():
        print(f"{len(top_10) - list(top_10.index).index(i):2d}. {row['collection_name'][:40]:40} | "
              f"Connections: {row['total_connections']:2d} | "
              f"Weighted Score: {row['degree_centrality_weighted']:.4f}")
    
    # Save graph structure
    graphml_path = OUTPUT_DIR / f"collection_graph_{timestamp}.graphml"
    nx.write_graphml(G, graphml_path)
    print(f"< Graph saved to: {graphml_path}")

def main():
    """Main analysis pipeline"""
    print("=� Starting NFT Collection Cluster Analysis")
    print("=" * 60)
    print("METHOD: Degree Centrality Analysis")
    print("OBJECTIVE: Identify collections with most shared holder relationships")
    print("=" * 60)
    
    # Setup
    setup_output_directory()
    
    # Load data
    collections_df = load_collection_data()
    collection_holders = load_holder_data()
    
    # Build graph
    G = build_similarity_graph(collections_df, collection_holders)
    
    # Analyze degree centrality
    results_df = analyze_degree_centrality(G, collections_df)
    
    # Save results
    save_results(results_df, G)
    
    print("\n Analysis complete!")
    print(f"=� Check {OUTPUT_DIR}/ for detailed results")

if __name__ == "__main__":
    main()