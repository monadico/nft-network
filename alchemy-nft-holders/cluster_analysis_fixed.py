#!/usr/bin/env python3
"""
NFT Collection Cluster Analysis using Graph Theory

This script analyzes relationships between NFT collections based on shared holders.
It represents collections as nodes in a graph and shared holders as weighted edges.

Analysis Methods:
1. DEGREE CENTRALITY - Measures direct connections to other collections
2. EIGENVECTOR CENTRALITY - Measures connections to other influential collections
"""

import sqlite3
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, Set, Tuple, List
from datetime import datetime

# Configuration
DB_PATH = Path('data/holders.sqlite')
OUTPUT_DIR = Path('analysis_results')
SIMILARITY_THRESHOLD = 0.01  # Minimum Jaccard similarity to create an edge

def setup_output_directory():
    """Create output directory for analysis results"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

def load_collection_data() -> pd.DataFrame:
    """
    Load collection metadata from the database
    
    Returns:
        DataFrame with collection_id, name, and owner_count
    """
    print("Loading collection data from database...")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get collection metadata
    collections_df = pd.read_sql_query("""
        SELECT collection_id, name, owner_count, processed_at
        FROM collections
        WHERE owner_count > 0
        ORDER BY owner_count DESC
    """, conn)
    
    conn.close()
    
    print(f"Loaded {len(collections_df)} collections")
    print(f"   Total holders range: {collections_df['owner_count'].min()} - {collections_df['owner_count'].max()}")
    
    return collections_df

def load_holder_data() -> Dict[str, Set[str]]:
    """
    Load holder data and organize by collection
    
    Returns:
        Dictionary mapping collection_id -> set of holder addresses
    """
    print("Loading holder data from database...")
    
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
    
    print(f"Loaded holder data for {len(collection_holders)} collections")
    print(f"   Total unique holder relationships: {len(holders_df)}")
    
    return collection_holders

def calculate_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Calculate Jaccard similarity coefficient between two sets
    
    Jaccard Index = |A intersection B| / |A union B|
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
    print("Building collection similarity graph...")
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
    
    print(f"Graph construction complete:")
    print(f"   Nodes (collections): {G.number_of_nodes()}")
    print(f"   Edges (relationships): {G.number_of_edges()}")
    print(f"   Edge density: {nx.density(G):.4f}")
    
    return G

# ==================================================================================
# DEGREE CENTRALITY ANALYSIS
# ==================================================================================

def analyze_degree_centrality(G: nx.Graph, collections_df: pd.DataFrame) -> pd.DataFrame:
    """
    DEGREE CENTRALITY ANALYSIS
    
    Degree centrality measures the importance of a node based on the number of connections.
    
    MATHEMATICAL DEFINITION:
    - Unweighted: C_D(v) = deg(v) / (n-1)
    - Weighted: C_D(v) = sum of w(u,v) for all neighbors u
    
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
    print("Computing Degree Centrality...")
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
    
    print(f"Degree centrality analysis complete")
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
# LOUVAIN COMMUNITY DETECTION
# ==================================================================================

def analyze_louvain_communities(G: nx.Graph, collections_df: pd.DataFrame) -> Tuple[Dict[str, int], pd.DataFrame]:
    """
    LOUVAIN COMMUNITY DETECTION ANALYSIS
    
    Louvain is a partition-based algorithm that maximizes modularity to identify communities.
    
    MATHEMATICAL DEFINITION:
    - Modularity Q = (1/2m) * sum[(A_ij - k_i*k_j/2m) * delta(c_i, c_j)]
    - Where A_ij = adjacency matrix, k_i = degree of node i, m = total edges
    - delta(c_i, c_j) = 1 if nodes i,j in same community, 0 otherwise
    
    ALGORITHM STEPS:
    1. Start with each node in its own community
    2. For each node, calculate modularity gain by moving to neighbor communities
    3. Move node to community that gives maximum modularity gain
    4. Repeat until no improvement possible
    5. Create super-nodes from communities and repeat
    
    INTERPRETATION:
    - Higher modularity = stronger community structure
    - Communities represent groups of collections with dense internal connections
    - Identifies "subcommunities" or "clusters" in the holder ecosystem
    - Each collection gets assigned a community ID
    
    BUSINESS MEANING:
    - Collections in same community likely share similar holder demographics
    - Could represent thematic groups (e.g., gaming NFTs, art NFTs, utility NFTs)
    - Useful for targeted marketing or partnership strategies
    
    Args:
        G: NetworkX graph with weighted edges
        collections_df: Collection metadata
        
    Returns:
        Tuple of (community_assignments_dict, results_dataframe)
    """
    print("Computing Louvain Community Detection...")
    print("   Method: Modularity maximization to identify subcommunities")
    print("   Interpretation: Groups collections with dense internal connections")
    
    try:
        # Import community detection library
        import community as community_louvain
    except ImportError:
        print("   ERROR: python-louvain library not installed")
        print("   Install with: pip install python-louvain")
        return {}, pd.DataFrame()
    
    # Run Louvain community detection
    partition = community_louvain.best_partition(G, weight='weight', random_state=42)
    
    # Calculate modularity score
    modularity = community_louvain.modularity(partition, G, weight='weight')
    
    # Analyze community statistics
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)
    
    num_communities = len(communities)
    community_sizes = [len(members) for members in communities.values()]
    
    print(f"Community detection complete:")
    print(f"   Number of communities found: {num_communities}")
    print(f"   Modularity score: {modularity:.4f}")
    print(f"   Community sizes: {sorted(community_sizes, reverse=True)}")
    print(f"   Largest community: {max(community_sizes)} collections")
    print(f"   Smallest community: {min(community_sizes)} collections")
    
    # Create detailed results DataFrame
    results = []
    for _, collection in collections_df.iterrows():
        collection_id = collection['collection_id']
        community_id = partition.get(collection_id, -1)  # -1 if not in graph
        
        # Calculate intra-community connections (connections within same community)
        intra_community_connections = 0
        inter_community_connections = 0
        intra_community_weight = 0
        inter_community_weight = 0
        
        if collection_id in G:
            for neighbor in G.neighbors(collection_id):
                edge_weight = G[collection_id][neighbor]['weight']
                
                if partition.get(neighbor, -1) == community_id:
                    intra_community_connections += 1
                    intra_community_weight += edge_weight
                else:
                    inter_community_connections += 1
                    inter_community_weight += edge_weight
        
        # Community cohesion score (ratio of internal to total connections)
        total_connections = intra_community_connections + inter_community_connections
        community_cohesion = intra_community_connections / total_connections if total_connections > 0 else 0
        
        results.append({
            'collection_id': collection_id,
            'collection_name': collection['name'],
            'holder_count': collection['owner_count'],
            'community_id': community_id,
            'community_size': len(communities.get(community_id, [])),
            'total_connections': total_connections,
            'intra_community_connections': intra_community_connections,
            'inter_community_connections': inter_community_connections,
            'community_cohesion_score': community_cohesion,
            'intra_community_weight': intra_community_weight,
            'inter_community_weight': inter_community_weight,
        })
    
    results_df = pd.DataFrame(results)
    
    # Sort by community ID, then by holder count within community
    results_df = results_df.sort_values(['community_id', 'holder_count'], ascending=[True, False])
    
    # Add community summary statistics
    community_summary = []
    for community_id in sorted(communities.keys()):
        community_collections = results_df[results_df['community_id'] == community_id]
        
        community_summary.append({
            'community_id': community_id,
            'community_size': len(community_collections),
            'total_holders': community_collections['holder_count'].sum(),
            'avg_holders': community_collections['holder_count'].mean(),
            'max_holders': community_collections['holder_count'].max(),
            'avg_cohesion': community_collections['community_cohesion_score'].mean(),
            'dominant_collections': ', '.join(community_collections.head(3)['collection_name'].tolist())
        })
    
    community_summary_df = pd.DataFrame(community_summary)
    community_summary_df = community_summary_df.sort_values('community_size', ascending=False)
    
    print("\nTop 5 Communities by Size:")
    for _, row in community_summary_df.head(5).iterrows():
        print(f"Community {row['community_id']:2d}: {row['community_size']:2d} collections | "
              f"Avg Holders: {row['avg_holders']:6.0f} | "
              f"Top: {row['dominant_collections']}")
    
    return partition, results_df, community_summary_df

# ==================================================================================
# GIRVAN-NEWMAN COMMUNITY DETECTION  
# ==================================================================================

def analyze_girvan_newman_communities(G: nx.Graph, collections_df: pd.DataFrame) -> Tuple[List[Set], pd.DataFrame, pd.DataFrame]:
    """
    GIRVAN-NEWMAN COMMUNITY DETECTION ANALYSIS
    
    Girvan-Newman is an edge-betweenness based algorithm that reveals hierarchical communities.
    
    MATHEMATICAL DEFINITION:
    - Edge betweenness: Number of shortest paths between all node pairs that pass through an edge
    - Algorithm iteratively removes edges with highest betweenness
    - Creates a dendrogram of community splits at different levels
    
    ALGORITHM STEPS:
    1. Calculate betweenness centrality for all edges
    2. Remove edge(s) with highest betweenness
    3. Recalculate betweenness for remaining edges  
    4. Repeat until no edges remain
    5. Choose optimal cut level using modularity score
    
    INTERPRETATION:
    - Reveals hierarchical structure of communities
    - High-betweenness edges are "bridges" between communities
    - Different hierarchy levels show different granularity of communities
    - Better for understanding nested community structures
    
    BUSINESS MEANING:
    - Identifies "bridge" collections that connect different holder segments
    - Shows how communities can be subdivided at different granularities
    - Useful for understanding market structure at multiple scales
    - Helps identify collections that serve as "gateways" between communities
    
    COMPARISON WITH LOUVAIN:
    - Louvain: Fast, single-level, modularity optimization
    - Girvan-Newman: Slower, hierarchical, edge-betweenness based
    - G-N better for understanding community hierarchy and bridges
    
    Args:
        G: NetworkX graph with weighted edges
        collections_df: Collection metadata
        
    Returns:
        Tuple of (best_communities, results_dataframe, level_analysis_df)
    """
    print("Computing Girvan-Newman Community Detection...")
    print("   Method: Edge-betweenness based hierarchical community detection")
    print("   Interpretation: Reveals hierarchical community structure")
    
    # Run Girvan-Newman algorithm (get iterator of community levels)
    communities_generator = nx.community.girvan_newman(G)
    
    # Analyze different levels of the hierarchy
    level_analysis = []
    best_modularity = -1
    best_communities = []
    best_level = 0
    
    print("   Analyzing hierarchy levels...")
    
    # Test up to 10 levels or until we have singleton communities
    max_levels = min(10, G.number_of_nodes() - 1)
    
    for level in range(max_levels):
        try:
            communities = next(communities_generator)
            communities_list = [frozenset(c) for c in communities]
            
            # Calculate modularity for this level
            modularity = nx.community.modularity(G, communities_list, weight='weight')
            
            # Count community sizes
            community_sizes = [len(c) for c in communities_list]
            num_communities = len(communities_list)
            
            level_analysis.append({
                'level': level + 1,
                'num_communities': num_communities,
                'modularity': modularity,
                'max_community_size': max(community_sizes),
                'min_community_size': min(community_sizes),
                'avg_community_size': sum(community_sizes) / num_communities,
                'community_sizes': sorted(community_sizes, reverse=True)
            })
            
            # Track best modularity
            if modularity > best_modularity:
                best_modularity = modularity
                best_communities = communities_list
                best_level = level + 1
            
            print(f"   Level {level + 1}: {num_communities} communities, modularity = {modularity:.4f}")
            
            # Stop if we have too many small communities
            if num_communities > G.number_of_nodes() // 2:
                break
                
        except StopIteration:
            break
    
    print(f"Girvan-Newman analysis complete:")
    print(f"   Best level: {best_level} (modularity = {best_modularity:.4f})")
    print(f"   Best configuration: {len(best_communities)} communities")
    
    # Create level analysis DataFrame
    level_analysis_df = pd.DataFrame(level_analysis)
    
    # Convert best communities to node assignment dictionary
    partition = {}
    for community_id, community in enumerate(best_communities):
        for node in community:
            partition[node] = community_id
    
    # Create detailed results DataFrame (similar to Louvain)
    results = []
    communities_dict = {}
    for community_id, community in enumerate(best_communities):
        communities_dict[community_id] = list(community)
    
    for _, collection in collections_df.iterrows():
        collection_id = collection['collection_id']
        community_id = partition.get(collection_id, -1)
        
        # Calculate intra-community connections
        intra_community_connections = 0
        inter_community_connections = 0
        intra_community_weight = 0
        inter_community_weight = 0
        
        if collection_id in G:
            for neighbor in G.neighbors(collection_id):
                edge_weight = G[collection_id][neighbor]['weight']
                
                if partition.get(neighbor, -1) == community_id:
                    intra_community_connections += 1
                    intra_community_weight += edge_weight
                else:
                    inter_community_connections += 1
                    inter_community_weight += edge_weight
        
        # Community cohesion score
        total_connections = intra_community_connections + inter_community_connections
        community_cohesion = intra_community_connections / total_connections if total_connections > 0 else 0
        
        # Calculate bridge score (ratio of inter-community connections)
        bridge_score = inter_community_connections / total_connections if total_connections > 0 else 0
        
        results.append({
            'collection_id': collection_id,
            'collection_name': collection['name'],
            'holder_count': collection['owner_count'],
            'gn_community_id': community_id,
            'gn_community_size': len(communities_dict.get(community_id, [])),
            'total_connections': total_connections,
            'intra_community_connections': intra_community_connections,
            'inter_community_connections': inter_community_connections,
            'gn_community_cohesion_score': community_cohesion,
            'gn_bridge_score': bridge_score,  # Unique to Girvan-Newman analysis
            'intra_community_weight': intra_community_weight,
            'inter_community_weight': inter_community_weight,
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['gn_community_id', 'holder_count'], ascending=[True, False])
    
    # Create community summary
    community_summary = []
    for community_id in sorted(communities_dict.keys()):
        community_collections = results_df[results_df['gn_community_id'] == community_id]
        
        community_summary.append({
            'gn_community_id': community_id,
            'gn_community_size': len(community_collections),
            'total_holders': community_collections['holder_count'].sum(),
            'avg_holders': community_collections['holder_count'].mean(),
            'max_holders': community_collections['holder_count'].max(),
            'avg_cohesion': community_collections['gn_community_cohesion_score'].mean(),
            'avg_bridge_score': community_collections['gn_bridge_score'].mean(),
            'dominant_collections': ', '.join(community_collections.head(3)['collection_name'].tolist())
        })
    
    community_summary_df = pd.DataFrame(community_summary)
    community_summary_df = community_summary_df.sort_values('gn_community_size', ascending=False)
    
    print("\nGirvan-Newman Communities (Best Level):")
    for _, row in community_summary_df.head(5).iterrows():
        print(f"Community {row['gn_community_id']:2d}: {row['gn_community_size']:2d} collections | "
              f"Avg Holders: {row['avg_holders']:6.0f} | "
              f"Bridge Score: {row['avg_bridge_score']:.3f} | "
              f"Top: {row['dominant_collections']}")
    
    return best_communities, results_df, community_summary_df, level_analysis_df

# ==================================================================================
# COMBINED ANALYSIS AND RESULTS
# ==================================================================================

def save_results(degree_results_df: pd.DataFrame, eigenvector_results_df: pd.DataFrame, 
                louvain_results_df: pd.DataFrame, community_summary_df: pd.DataFrame,
                gn_results_df: pd.DataFrame, gn_summary_df: pd.DataFrame, 
                gn_hierarchy_df: pd.DataFrame, G: nx.Graph):
    """Save analysis results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save degree centrality results
    degree_csv_path = OUTPUT_DIR / f"degree_centrality_analysis_{timestamp}.csv"
    degree_results_df.to_csv(degree_csv_path, index=False)
    print(f"Results saved to: {degree_csv_path}")
    
    # Save eigenvector centrality results
    eigen_csv_path = OUTPUT_DIR / f"eigenvector_centrality_analysis_{timestamp}.csv"
    eigenvector_results_df.to_csv(eigen_csv_path, index=False)
    print(f"Results saved to: {eigen_csv_path}")
    
    # Save Louvain community detection results
    louvain_csv_path = OUTPUT_DIR / f"louvain_communities_analysis_{timestamp}.csv"
    louvain_results_df.to_csv(louvain_csv_path, index=False)
    print(f"Results saved to: {louvain_csv_path}")
    
    # Save community summary
    community_summary_csv_path = OUTPUT_DIR / f"louvain_community_summary_{timestamp}.csv"
    community_summary_df.to_csv(community_summary_csv_path, index=False)
    print(f"Louvain community summary saved to: {community_summary_csv_path}")
    
    # Save Girvan-Newman community detection results
    gn_csv_path = OUTPUT_DIR / f"girvan_newman_communities_{timestamp}.csv"
    gn_results_df.to_csv(gn_csv_path, index=False)
    print(f"Results saved to: {gn_csv_path}")
    
    # Save Girvan-Newman community summary
    gn_summary_csv_path = OUTPUT_DIR / f"girvan_newman_summary_{timestamp}.csv"
    gn_summary_df.to_csv(gn_summary_csv_path, index=False)
    print(f"Girvan-Newman summary saved to: {gn_summary_csv_path}")
    
    # Save hierarchy analysis
    gn_hierarchy_csv_path = OUTPUT_DIR / f"girvan_newman_hierarchy_{timestamp}.csv"
    gn_hierarchy_df.to_csv(gn_hierarchy_csv_path, index=False)
    print(f"Girvan-Newman hierarchy analysis saved to: {gn_hierarchy_csv_path}")
    
    # Create combined comparison
    combined_df = degree_results_df[['collection_id', 'collection_name', 'holder_count', 
                                   'degree_centrality_weighted', 'total_connections']].copy()
    combined_df = combined_df.merge(
        eigenvector_results_df[['collection_id', 'eigenvector_centrality_weighted', 'neighbor_influence_score']], 
        on='collection_id', how='left'
    )
    combined_df = combined_df.merge(
        louvain_results_df[['collection_id', 'community_id', 'community_size', 'community_cohesion_score']], 
        on='collection_id', how='left'
    )
    combined_df = combined_df.merge(
        gn_results_df[['collection_id', 'gn_community_id', 'gn_community_size', 'gn_bridge_score']], 
        on='collection_id', how='left'
    )
    combined_df['centrality_rank_degree'] = combined_df['degree_centrality_weighted'].rank(ascending=False)
    combined_df['centrality_rank_eigenvector'] = combined_df['eigenvector_centrality_weighted'].rank(ascending=False)
    combined_df['rank_difference'] = combined_df['centrality_rank_degree'] - combined_df['centrality_rank_eigenvector']
    
    # Add community agreement indicator (do Louvain and G-N assign to same community size?)
    combined_df['community_agreement'] = (combined_df['community_size'] == combined_df['gn_community_size']).astype(int)
    
    combined_csv_path = OUTPUT_DIR / f"combined_centrality_analysis_{timestamp}.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Combined results saved to: {combined_csv_path}")
    
    # Print top 10 summaries
    print("\n" + "="*80)
    print("TOP 10 COLLECTIONS BY DEGREE CENTRALITY:")
    print("="*80)
    degree_top_10 = degree_results_df.head(10)
    for i, (_, row) in enumerate(degree_top_10.iterrows(), 1):
        print(f"{i:2d}. {row['collection_name'][:40]:40} | "
              f"Connections: {row['total_connections']:2d} | "
              f"Weighted Score: {row['degree_centrality_weighted']:.4f}")
    
    print("\n" + "="*80)
    print("TOP 10 COLLECTIONS BY EIGENVECTOR CENTRALITY:")
    print("="*80)
    eigen_top_10 = eigenvector_results_df.head(10)
    for i, (_, row) in enumerate(eigen_top_10.iterrows(), 1):
        print(f"{i:2d}. {row['collection_name'][:40]:40} | "
              f"Connections: {row['total_connections']:2d} | "
              f"Eigen Score: {row['eigenvector_centrality_weighted']:.4f}")
    
    print("\n" + "="*80)
    print("COMMUNITY DETECTION COMPARISON:")
    print("="*80)
    
    # Community method comparison
    louvain_communities = len(community_summary_df)
    gn_communities = len(gn_summary_df)
    agreement_rate = combined_df['community_agreement'].mean()
    
    print(f"LOUVAIN:       {louvain_communities} communities found")
    print(f"GIRVAN-NEWMAN: {gn_communities} communities found")
    print(f"Community size agreement rate: {agreement_rate:.2%}")
    
    print(f"\nLouvain - Largest community: {community_summary_df['community_size'].max()} collections")
    print(f"G-N - Largest community:     {gn_summary_df['gn_community_size'].max()} collections")
    
    # Show top bridge collections (high G-N bridge score)
    top_bridges = combined_df.nlargest(5, 'gn_bridge_score')
    print(f"\nTOP 5 BRIDGE COLLECTIONS (Connect Different Communities):")
    for _, row in top_bridges.iterrows():
        print(f"{row['collection_name'][:35]:35} | Bridge Score: {row['gn_bridge_score']:.3f} | "
              f"Louvain: C{int(row['community_id']) if pd.notna(row['community_id']) else '?'} | "
              f"G-N: C{int(row['gn_community_id']) if pd.notna(row['gn_community_id']) else '?'}")
    
    print("\n" + "="*80)
    print("BIGGEST RANKING DIFFERENCES (Degree vs Eigenvector):")
    print("="*80)
    biggest_diffs = combined_df.reindex(combined_df['rank_difference'].abs().nlargest(5).index)
    for _, row in biggest_diffs.iterrows():
        direction = "Higher in Degree" if row['rank_difference'] < 0 else "Higher in Eigenvector"
        community_info = f"Community {int(row['community_id'])}" if pd.notna(row['community_id']) else "No Community"
        print(f"{row['collection_name'][:35]:35} | "
              f"Degree: {int(row['centrality_rank_degree']):2d} | "
              f"Eigen: {int(row['centrality_rank_eigenvector']):2d} | "
              f"Diff: {abs(int(row['rank_difference'])):2d} | "
              f"{community_info}")
    
    # Save graph structure
    graphml_path = OUTPUT_DIR / f"collection_graph_{timestamp}.graphml"
    nx.write_graphml(G, graphml_path)
    print(f"\nGraph saved to: {graphml_path}")

def main():
    """Main analysis pipeline"""
    print("Starting NFT Collection Cluster Analysis")
    print("=" * 60)
    print("METHODS: Degree Centrality + Eigenvector Centrality Analysis")
    print("OBJECTIVE: Identify collections with most shared holder relationships")
    print("=" * 60)
    
    # Setup
    setup_output_directory()
    
    # Load data
    collections_df = load_collection_data()
    collection_holders = load_holder_data()
    
    # Build graph
    G = build_similarity_graph(collections_df, collection_holders)
    
    print("\n" + "="*80)
    print("DEGREE CENTRALITY ANALYSIS")
    print("="*80)
    # Analyze degree centrality
    degree_results_df = analyze_degree_centrality(G, collections_df)
    
    print("\n" + "="*80) 
    print("EIGENVECTOR CENTRALITY ANALYSIS")
    print("="*80)
    # Analyze eigenvector centrality
    eigenvector_results_df = analyze_eigenvector_centrality(G, collections_df)
    
    print("\n" + "="*80)
    print("LOUVAIN COMMUNITY DETECTION")
    print("="*80)
    # Analyze Louvain communities
    partition, louvain_results_df, community_summary_df = analyze_louvain_communities(G, collections_df)
    
    print("\n" + "="*80)
    print("GIRVAN-NEWMAN COMMUNITY DETECTION")
    print("="*80)
    # Analyze Girvan-Newman communities
    gn_communities, gn_results_df, gn_summary_df, gn_hierarchy_df = analyze_girvan_newman_communities(G, collections_df)
    
    # Save results
    save_results(degree_results_df, eigenvector_results_df, louvain_results_df, community_summary_df,
                gn_results_df, gn_summary_df, gn_hierarchy_df, G)
    
    print("\nAnalysis complete!")
    print(f"Check {OUTPUT_DIR}/ for detailed results")

if __name__ == "__main__":
    main()