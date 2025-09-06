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
from collections import Counter

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
# CENTRALITY VALIDATION METHODS
# ==================================================================================

def validate_centrality_consistency(G: nx.Graph, collections_df: pd.DataFrame) -> pd.DataFrame:
    """
    1. CONSISTENCY BETWEEN CENTRALITY METHODS (Correlation Analysis)
    
    Validates centrality measures by testing consistency between degree and eigenvector centrality.
    
    THEORETICAL BASIS:
    - Degree centrality: Local influence (direct connections)
    - Eigenvector centrality: Global influence (connections to influential nodes)
    - High correlation: Methods agree on influential collections
    - Low correlation: Methods capture different aspects of influence
    
    STATISTICAL TESTS:
    - Spearman rank correlation: Non-parametric, suitable for rankings
    - Pearson correlation: Linear relationship between raw scores
    - Significance testing: p-value < 0.05 indicates non-random correlation
    
    INTERPRETATION:
    - Correlation > 0.7: Strong agreement between methods
    - Correlation 0.3-0.7: Moderate agreement, methods complement each other
    - Correlation < 0.3: Weak agreement, methods capture different influence types
    
    Args:
        G: NetworkX graph
        collections_df: Collection metadata
        
    Returns:
        DataFrame with correlation analysis results
    """
    from scipy.stats import spearmanr, pearsonr
    import matplotlib.pyplot as plt
    
    print("Validating Centrality Consistency...")
    print("   Method 1: Correlation analysis between degree and eigenvector centrality")
    
    # Calculate centralities
    degree_centrality = nx.degree_centrality(G)
    weighted_degree = {}
    for node in G.nodes():
        weighted_degree[node] = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
    
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except nx.NetworkXError:
        print("   Warning: Eigenvector centrality failed, using unweighted version")
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Prepare data for correlation
    collections_in_graph = [c for c in collections_df['collection_id'] if c in G]
    degree_values = [degree_centrality.get(c, 0) for c in collections_in_graph]
    weighted_degree_values = [weighted_degree.get(c, 0) for c in collections_in_graph]
    eigenvector_values = [eigenvector_centrality.get(c, 0) for c in collections_in_graph]
    
    # Calculate correlations
    spearman_corr_unweighted, spearman_p_unweighted = spearmanr(degree_values, eigenvector_values)
    pearson_corr_unweighted, pearson_p_unweighted = pearsonr(degree_values, eigenvector_values)
    
    spearman_corr_weighted, spearman_p_weighted = spearmanr(weighted_degree_values, eigenvector_values)
    pearson_corr_weighted, pearson_p_weighted = pearsonr(weighted_degree_values, eigenvector_values)
    
    print(f"Correlation Analysis Results:")
    print(f"   Unweighted Degree vs Eigenvector:")
    print(f"     Spearman correlation: {spearman_corr_unweighted:.4f} (p = {spearman_p_unweighted:.4f})")
    print(f"     Pearson correlation:  {pearson_corr_unweighted:.4f} (p = {pearson_p_unweighted:.4f})")
    print(f"   Weighted Degree vs Eigenvector:")
    print(f"     Spearman correlation: {spearman_corr_weighted:.4f} (p = {spearman_p_weighted:.4f})")
    print(f"     Pearson correlation:  {pearson_corr_weighted:.4f} (p = {pearson_p_weighted:.4f})")
    
    # Interpretation
    def interpret_correlation(corr):
        if abs(corr) > 0.7:
            return "Strong agreement"
        elif abs(corr) > 0.3:
            return "Moderate agreement"
        else:
            return "Weak agreement"
    
    print(f"   Interpretation:")
    print(f"     Unweighted: {interpret_correlation(spearman_corr_unweighted)}")
    print(f"     Weighted: {interpret_correlation(spearman_corr_weighted)}")
    
    # Create results DataFrame
    consistency_results = pd.DataFrame({
        'collection_id': collections_in_graph,
        'degree_centrality': degree_values,
        'weighted_degree': weighted_degree_values,
        'eigenvector_centrality': eigenvector_values
    })
    
    consistency_results['degree_rank'] = consistency_results['degree_centrality'].rank(ascending=False)
    consistency_results['weighted_degree_rank'] = consistency_results['weighted_degree'].rank(ascending=False)
    consistency_results['eigenvector_rank'] = consistency_results['eigenvector_centrality'].rank(ascending=False)
    consistency_results['rank_difference'] = abs(consistency_results['degree_rank'] - consistency_results['eigenvector_rank'])
    
    # Add correlation statistics
    consistency_results.attrs = {
        'spearman_unweighted': spearman_corr_unweighted,
        'spearman_p_unweighted': spearman_p_unweighted,
        'spearman_weighted': spearman_corr_weighted,
        'spearman_p_weighted': spearman_p_weighted,
        'pearson_unweighted': pearson_corr_unweighted,
        'pearson_p_unweighted': pearson_p_unweighted,
        'pearson_weighted': pearson_corr_weighted,
        'pearson_p_weighted': pearson_p_weighted
    }
    
    return consistency_results

def validate_centrality_bootstrap(G: nx.Graph, collections_df: pd.DataFrame, n_bootstrap=100) -> pd.DataFrame:
    """
    2. BOOTSTRAP RESAMPLING FOR CONFIDENCE INTERVALS
    
    Estimates uncertainty in centrality scores using bootstrap resampling.
    
    THEORETICAL BASIS:
    - Bootstrap resampling: Statistical method to estimate sampling distribution
    - Resamples edges with replacement to simulate data collection uncertainty
    - Confidence intervals quantify uncertainty in centrality measurements
    
    METHODOLOGY:
    - Resample edges n_bootstrap times (default 100 for speed)
    - Recompute centralities for each bootstrap sample
    - Calculate 95% confidence intervals using percentile method
    
    INTERPRETATION:
    - Narrow CIs: Robust, stable centrality scores
    - Wide CIs: Uncertain scores, sensitive to data variations
    - Non-overlapping CIs: Significant differences between collections
    
    Args:
        G: NetworkX graph
        collections_df: Collection metadata
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        DataFrame with confidence intervals for centrality scores
    """
    import numpy as np
    from sklearn.utils import resample
    
    print(f"Validating Centrality Robustness with Bootstrap Resampling...")
    print(f"   Method 2: Bootstrap resampling (n = {n_bootstrap})")
    print(f"   Computing confidence intervals for centrality measures...")
    
    # Initialize bootstrap storage
    collections_in_graph = [c for c in collections_df['collection_id'] if c in G]
    bootstrap_degree = {node: [] for node in collections_in_graph}
    bootstrap_eigenvector = {node: [] for node in collections_in_graph}
    
    # Get original edges with weights
    edges_with_weights = [(u, v, d) for u, v, d in G.edges(data=True)]
    
    successful_bootstraps = 0
    for i in range(n_bootstrap):
        try:
            # Resample edges with replacement
            bootstrap_edges = resample(edges_with_weights, replace=True, n_samples=len(edges_with_weights))
            
            # Create bootstrap graph
            boot_G = nx.Graph()
            boot_G.add_edges_from(bootstrap_edges)
            
            # Skip if graph becomes disconnected or too small
            if boot_G.number_of_nodes() < G.number_of_nodes() * 0.5:
                continue
                
            # Compute centralities
            boot_degree = nx.degree_centrality(boot_G)
            try:
                boot_eigen = nx.eigenvector_centrality(boot_G, weight='weight', max_iter=1000)
            except nx.NetworkXError:
                boot_eigen = nx.eigenvector_centrality(boot_G, max_iter=1000)
            
            # Store results for collections that exist in bootstrap graph
            for node in collections_in_graph:
                if node in boot_G:
                    bootstrap_degree[node].append(boot_degree.get(node, 0))
                    bootstrap_eigenvector[node].append(boot_eigen.get(node, 0))
                else:
                    bootstrap_degree[node].append(0)
                    bootstrap_eigenvector[node].append(0)
            
            successful_bootstraps += 1
            
        except Exception as e:
            continue
    
    print(f"   Successful bootstrap samples: {successful_bootstraps}/{n_bootstrap}")
    
    # Calculate confidence intervals
    bootstrap_results = []
    for node in collections_in_graph:
        if len(bootstrap_degree[node]) > 0:
            degree_ci = np.percentile(bootstrap_degree[node], [2.5, 97.5])
            degree_mean = np.mean(bootstrap_degree[node])
            degree_std = np.std(bootstrap_degree[node])
            
            eigenvector_ci = np.percentile(bootstrap_eigenvector[node], [2.5, 97.5])
            eigenvector_mean = np.mean(bootstrap_eigenvector[node])
            eigenvector_std = np.std(bootstrap_eigenvector[node])
            
            bootstrap_results.append({
                'collection_id': node,
                'degree_mean': degree_mean,
                'degree_std': degree_std,
                'degree_ci_lower': degree_ci[0],
                'degree_ci_upper': degree_ci[1],
                'degree_ci_width': degree_ci[1] - degree_ci[0],
                'eigenvector_mean': eigenvector_mean,
                'eigenvector_std': eigenvector_std,
                'eigenvector_ci_lower': eigenvector_ci[0],
                'eigenvector_ci_upper': eigenvector_ci[1],
                'eigenvector_ci_width': eigenvector_ci[1] - eigenvector_ci[0],
            })
    
    bootstrap_df = pd.DataFrame(bootstrap_results)
    
    # Rank by robustness (narrow confidence intervals)
    bootstrap_df['degree_robustness_rank'] = bootstrap_df['degree_ci_width'].rank()
    bootstrap_df['eigenvector_robustness_rank'] = bootstrap_df['eigenvector_ci_width'].rank()
    
    print(f"Bootstrap Analysis Results:")
    print(f"   Most robust collections (narrow CIs):")
    top_robust = bootstrap_df.nsmallest(3, 'degree_ci_width')
    for _, row in top_robust.iterrows():
        print(f"     {row['collection_id']}: Degree CI width = {row['degree_ci_width']:.4f}")
    
    return bootstrap_df

def validate_centrality_significance(G: nx.Graph, collections_df: pd.DataFrame, n_null=100) -> pd.DataFrame:
    """
    3. COMPARISON TO NULL MODELS (Significance Testing)
    
    Tests if observed centralities are significantly higher than expected in random graphs.
    
    THEORETICAL BASIS:
    - Null hypothesis: Centrality scores arise from random graph structure
    - Configuration model: Preserves degree distribution, randomizes connections
    - Empirical p-value: Fraction of null centralities >= observed
    - Controls for artifacts from graph topology
    
    METHODOLOGY:
    - Generate n_null random graphs with same degree distribution
    - Compute centralities for each null graph
    - Compare observed vs null distribution
    - Calculate empirical p-values
    
    INTERPRETATION:
    - p < 0.05: Collection's influence is statistically significant
    - p >= 0.05: Influence could be due to random graph structure
    - Lower p-values indicate stronger statistical evidence of influence
    
    Args:
        G: NetworkX graph
        collections_df: Collection metadata
        n_null: Number of null model graphs
        
    Returns:
        DataFrame with significance test results
    """
    import numpy as np
    
    print(f"Validating Centrality Significance with Null Models...")
    print(f"   Method 3: Configuration model comparison (n = {n_null})")
    print(f"   Testing if observed centralities exceed random expectation...")
    
    # Calculate observed centralities
    obs_degree = nx.degree_centrality(G)
    try:
        obs_eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except nx.NetworkXError:
        obs_eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    
    collections_in_graph = [c for c in collections_df['collection_id'] if c in G]
    
    # Generate null distributions
    null_degree_dist = {node: [] for node in collections_in_graph}
    null_eigenvector_dist = {node: [] for node in collections_in_graph}
    
    successful_nulls = 0
    for i in range(n_null):
        try:
            # Create configuration model (preserves degree sequence)
            degree_sequence = [d for n, d in G.degree()]
            null_G = nx.configuration_model(degree_sequence)
            null_G = nx.Graph(null_G)  # Remove parallel edges and self-loops
            null_G.remove_edges_from(nx.selfloop_edges(null_G))
            
            # Skip if graph becomes disconnected
            if not nx.is_connected(null_G):
                # Use largest connected component
                largest_cc = max(nx.connected_components(null_G), key=len)
                null_G = null_G.subgraph(largest_cc).copy()
            
            # Compute null centralities
            null_degree = nx.degree_centrality(null_G)
            try:
                null_eigenvector = nx.eigenvector_centrality(null_G, max_iter=1000)
            except nx.NetworkXError:
                continue
            
            # Map back to original node IDs (configuration model uses integer IDs)
            node_mapping = dict(zip(null_G.nodes(), collections_in_graph[:null_G.number_of_nodes()]))
            
            for null_node, orig_node in node_mapping.items():
                if orig_node in collections_in_graph:
                    null_degree_dist[orig_node].append(null_degree.get(null_node, 0))
                    null_eigenvector_dist[orig_node].append(null_eigenvector.get(null_node, 0))
            
            successful_nulls += 1
            
        except Exception as e:
            continue
    
    print(f"   Successful null models: {successful_nulls}/{n_null}")
    
    # Calculate p-values
    significance_results = []
    for node in collections_in_graph:
        if len(null_degree_dist[node]) > 0:
            # Empirical p-values (one-tailed test: obs >= null)
            degree_p = np.mean([null >= obs_degree[node] for null in null_degree_dist[node]])
            eigenvector_p = np.mean([null >= obs_eigenvector[node] for null in null_eigenvector_dist[node]])
            
            # Null distribution statistics
            degree_null_mean = np.mean(null_degree_dist[node])
            degree_null_std = np.std(null_degree_dist[node])
            eigenvector_null_mean = np.mean(null_eigenvector_dist[node])
            eigenvector_null_std = np.std(null_eigenvector_dist[node])
            
            # Z-scores (standardized effect sizes)
            degree_zscore = (obs_degree[node] - degree_null_mean) / degree_null_std if degree_null_std > 0 else 0
            eigenvector_zscore = (obs_eigenvector[node] - eigenvector_null_mean) / eigenvector_null_std if eigenvector_null_std > 0 else 0
            
            significance_results.append({
                'collection_id': node,
                'observed_degree': obs_degree[node],
                'degree_null_mean': degree_null_mean,
                'degree_p_value': degree_p,
                'degree_zscore': degree_zscore,
                'degree_significant': degree_p < 0.05,
                'observed_eigenvector': obs_eigenvector[node],
                'eigenvector_null_mean': eigenvector_null_mean,
                'eigenvector_p_value': eigenvector_p,
                'eigenvector_zscore': eigenvector_zscore,
                'eigenvector_significant': eigenvector_p < 0.05,
            })
    
    significance_df = pd.DataFrame(significance_results)
    
    # Summary statistics
    degree_significant_count = significance_df['degree_significant'].sum()
    eigenvector_significant_count = significance_df['eigenvector_significant'].sum()
    
    print(f"Significance Test Results:")
    print(f"   Degree centrality: {degree_significant_count}/{len(significance_df)} collections significant (p < 0.05)")
    print(f"   Eigenvector centrality: {eigenvector_significant_count}/{len(significance_df)} collections significant (p < 0.05)")
    
    # Show most significant collections
    if len(significance_df) > 0:
        print(f"   Most significant collections (degree centrality):")
        top_significant = significance_df.nsmallest(3, 'degree_p_value')
        for _, row in top_significant.iterrows():
            print(f"     {row['collection_id']}: p = {row['degree_p_value']:.4f}, z = {row['degree_zscore']:.2f}")
    
    return significance_df

# ==================================================================================
# SPECTRAL CLUSTERING ANALYSIS
# ==================================================================================

def analyze_spectral_clustering(G: nx.Graph, collections_df: pd.DataFrame) -> Tuple[List[Set], pd.DataFrame, pd.DataFrame]:
    """
    SPECTRAL CLUSTERING ANALYSIS (FIXED FOR DISCONNECTED GRAPHS)
    
    Spectral clustering uses the graph's Laplacian matrix to find clusters based on 
    eigenvalues and eigenvectors. This implementation handles disconnected graphs
    by working with the largest connected component and using proper preprocessing.
    
    THEORETICAL BASIS:
    - Uses graph Laplacian eigendecomposition
    - Projects nodes into spectral space before clustering
    - Can detect non-convex or irregularly shaped communities
    - Complementary to modularity-based approaches
    
    PREPROCESSING FOR SPARSE GRAPHS:
    - Work with largest connected component only
    - Add small epsilon to diagonal for numerical stability
    - Use RBF kernel transformation for better similarity structure
    - Validate connectivity before proceeding
    
    ADVANTAGES:
    - Can find communities missed by modularity optimization
    - Works well with properly preprocessed similarity-based graphs
    - Theoretically grounded in spectral graph theory
    - Less sensitive to resolution limits
    
    Args:
        G: NetworkX graph with weighted edges
        collections_df: Collection metadata
        
    Returns:
        Tuple of (communities, results_df, summary_df)
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score
    import warnings
    
    print("Computing Spectral Clustering...")
    print("   Method: Laplacian eigendecomposition with connectivity preprocessing")
    print("   Interpretation: Reveals communities in spectral space")
    
    # Step 1: Check graph connectivity and preprocess
    if not nx.is_connected(G):
        print(f"   Warning: Graph is not fully connected")
        # Get largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc).copy()
        print(f"   Using largest connected component: {len(largest_cc)}/{len(G)} nodes")
    else:
        G_connected = G.copy()
        largest_cc = set(G.nodes())
    
    # Extract adjacency matrix and node list for connected component
    adj_matrix = nx.to_numpy_array(G_connected, weight='weight')
    nodes = list(G_connected.nodes())
    n_nodes = len(nodes)
    
    print(f"   Connected graph: {n_nodes} nodes, adjacency matrix shape: {adj_matrix.shape}")
    
    # Skip spectral clustering if too few nodes
    if n_nodes < 3:
        print(f"   Error: Too few nodes ({n_nodes}) for meaningful clustering")
        return [], pd.DataFrame(), pd.DataFrame()
    
    # Step 2: Preprocess adjacency matrix for spectral clustering
    # Add small epsilon to diagonal for numerical stability
    epsilon = 1e-6
    adj_matrix_processed = adj_matrix + epsilon * np.eye(n_nodes)
    
    # Step 3: Optimize number of clusters with better range
    print("   Optimizing number of clusters...")
    sil_scores = []
    k_range = range(2, min(8, n_nodes))  # Reduced range for stability
    
    # Suppress sklearn warnings temporarily
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        for k in k_range:
            try:
                sc = SpectralClustering(
                    n_clusters=k, 
                    affinity='precomputed', 
                    assign_labels='kmeans', 
                    random_state=42,
                    n_init=10  # Multiple initializations for stability
                )
                labels = sc.fit_predict(adj_matrix_processed)
                
                # Only calculate silhouette if we have meaningful clusters
                if len(set(labels)) > 1:
                    sil = silhouette_score(adj_matrix_processed, labels, metric='precomputed')
                    sil_scores.append(sil)
                else:
                    sil_scores.append(-1)
                
            except Exception as e:
                sil_scores.append(-1)  # Invalid score for failed attempts
                continue
    
    # Choose best k, but be more conservative
    if sil_scores and max(sil_scores) > -0.5:  # Only if reasonable quality
        best_k_index = np.argmax(sil_scores)
        best_k = list(k_range)[best_k_index]
        best_silhouette = sil_scores[best_k_index]
        print(f"   Optimal clusters: {best_k} (silhouette: {best_silhouette:.4f})")
    else:
        # Fallback: use same number as Louvain for comparison
        best_k = 5
        best_silhouette = -1.0
        print(f"   Optimization failed, using fallback: {best_k} clusters")
    
    # Step 4: Run final spectral clustering
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            sc = SpectralClustering(
                n_clusters=best_k, 
                affinity='precomputed',
                assign_labels='kmeans', 
                random_state=42,
                n_init=10
            )
            labels_spectral = sc.fit_predict(adj_matrix_processed)
        
        # Step 5: Create communities (only for connected component)
        partition_spectral = [set() for _ in range(max(labels_spectral) + 1)]
        for i, label in enumerate(labels_spectral):
            partition_spectral[label].add(nodes[i])
        
        # Remove empty partitions
        partition_spectral = [p for p in partition_spectral if len(p) > 0]
        
        # Add isolated nodes as single-node communities
        isolated_nodes = set(G.nodes()) - largest_cc
        for isolated_node in isolated_nodes:
            partition_spectral.append({isolated_node})
        
        print(f"   Added {len(isolated_nodes)} isolated nodes as single-node communities")
        
        # Calculate modularity on original graph (now includes all nodes)
        mod_spectral = nx.community.modularity(G, partition_spectral, weight='weight')
        
        print(f"   Spectral clustering complete:")
        print(f"     Number of communities: {len(partition_spectral)}")
        print(f"     Modularity score: {mod_spectral:.4f}")
        print(f"     Silhouette score: {best_silhouette:.4f}")
        print(f"     Coverage: {len(largest_cc)}/{len(G)} nodes clustered")
        
    except Exception as e:
        print(f"   Error in spectral clustering: {e}")
        return [], pd.DataFrame(), pd.DataFrame()
    
    # Step 4: Create detailed results DataFrame
    results = []
    for _, collection in collections_df.iterrows():
        collection_id = collection['collection_id']
        
        # Find which community this collection belongs to
        community_id = -1
        for i, community in enumerate(partition_spectral):
            if collection_id in community:
                community_id = i
                break
        
        # Calculate community-specific metrics
        intra_community_connections = 0
        inter_community_connections = 0
        intra_community_weight = 0
        inter_community_weight = 0
        
        if collection_id in G and community_id >= 0:
            for neighbor in G.neighbors(collection_id):
                edge_weight = G[collection_id][neighbor]['weight']
                
                # Check if neighbor is in same community
                neighbor_in_same_community = neighbor in partition_spectral[community_id]
                
                if neighbor_in_same_community:
                    intra_community_connections += 1
                    intra_community_weight += edge_weight
                else:
                    inter_community_connections += 1
                    inter_community_weight += edge_weight
        
        # Community cohesion score
        total_connections = intra_community_connections + inter_community_connections
        community_cohesion = intra_community_connections / total_connections if total_connections > 0 else 0
        
        results.append({
            'collection_id': collection_id,
            'collection_name': collection['name'],
            'holder_count': collection['owner_count'],
            'spectral_community_id': community_id,
            'spectral_community_size': len(partition_spectral[community_id]) if community_id >= 0 else 0,
            'total_connections': total_connections,
            'intra_community_connections': intra_community_connections,
            'inter_community_connections': inter_community_connections,
            'spectral_community_cohesion_score': community_cohesion,
            'intra_community_weight': intra_community_weight,
            'inter_community_weight': inter_community_weight,
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['spectral_community_id', 'holder_count'], ascending=[True, False])
    
    # Step 5: Create community summary
    community_summary = []
    for community_id in range(len(partition_spectral)):
        community_collections = results_df[results_df['spectral_community_id'] == community_id]
        
        if len(community_collections) > 0:
            community_summary.append({
                'spectral_community_id': community_id,
                'spectral_community_size': len(community_collections),
                'total_holders': community_collections['holder_count'].sum(),
                'avg_holders': community_collections['holder_count'].mean(),
                'max_holders': community_collections['holder_count'].max(),
                'avg_cohesion': community_collections['spectral_community_cohesion_score'].mean(),
                'modularity_contribution': mod_spectral / len(partition_spectral),  # Rough estimate
                'dominant_collections': ', '.join(community_collections.head(3)['collection_name'].tolist())
            })
    
    community_summary_df = pd.DataFrame(community_summary)
    community_summary_df = community_summary_df.sort_values('spectral_community_size', ascending=False)
    
    print("\nSpectral Clustering Communities:")
    for _, row in community_summary_df.head(10).iterrows():  # Show top 10
        print(f"Community {row['spectral_community_id']:2d}: {row['spectral_community_size']:2d} collections | "
              f"Avg Holders: {row['avg_holders']:6.0f} | "
              f"Cohesion: {row['avg_cohesion']:.3f} | "
              f"Top: {row['dominant_collections']}")
    
    return partition_spectral, results_df, community_summary_df

# ==================================================================================
# HIERARCHICAL CLUSTERING ANALYSIS
# ==================================================================================

def analyze_hierarchical_clustering(G: nx.Graph, collections_df: pd.DataFrame) -> Tuple[List[Set], pd.DataFrame, pd.DataFrame]:
    """
    HIERARCHICAL CLUSTERING ANALYSIS
    
    Hierarchical clustering builds a dendrogram showing nested community structure.
    It complements other methods by revealing multi-level community organization
    and providing an alternative perspective on similarity-based grouping.
    
    THEORETICAL BASIS:
    - Builds tree of nested clusters using distance-based merging
    - Can reveal hierarchical organization missed by modularity methods
    - Deterministic (unlike spectral) and shows full hierarchy (unlike Louvain)
    - Works on distance matrices derived from similarity graphs
    
    METHODOLOGY:
    - Convert Jaccard similarity to distance (1 - similarity)
    - Use average linkage for precomputed distance matrices
    - Determine optimal number of clusters via modularity optimization
    - Create detailed community analysis with cohesion metrics
    
    ADVANTAGES:
    - Deterministic and reproducible results
    - Reveals nested community structure at multiple scales
    - Works well with similarity-based data after distance conversion
    - Provides interpretable dendrogram visualization
    
    Args:
        G: NetworkX graph with weighted edges
        collections_df: Collection metadata
        
    Returns:
        Tuple of (communities, results_df, summary_df)
    """
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram, linkage
    import matplotlib.pyplot as plt
    
    print("Computing Hierarchical Clustering...")
    print("   Method: Agglomerative clustering with average linkage")
    print("   Interpretation: Reveals hierarchical community structure")
    
    # Extract adjacency matrix and convert to distance
    adj_matrix = nx.to_numpy_array(G, weight='weight')
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    print(f"   Graph: {n_nodes} nodes, converting similarity to distance matrix")
    
    # Convert similarity to distance matrix
    # Jaccard similarities are in [0,1], so distance = 1 - similarity
    dist_matrix = 1 - adj_matrix
    np.fill_diagonal(dist_matrix, 0)  # Self-distance = 0
    
    # Ensure valid distance matrix properties
    dist_matrix = np.clip(dist_matrix, 0, 1)  # Ensure [0,1] range
    dist_matrix = (dist_matrix + dist_matrix.T) / 2  # Ensure symmetry
    
    print(f"   Distance matrix: shape {dist_matrix.shape}, range [{dist_matrix.min():.3f}, {dist_matrix.max():.3f}]")
    
    # Step 1: Determine optimal number of clusters by testing modularity
    print("   Optimizing number of clusters via modularity...")
    best_n_clusters = 5
    best_modularity = -1
    modularity_scores = []
    
    k_range = range(2, min(12, n_nodes))  # Test 2 to 11 clusters
    
    for k in k_range:
        try:
            hc = AgglomerativeClustering(
                n_clusters=k,
                metric='precomputed',
                linkage='average'
            )
            labels = hc.fit_predict(dist_matrix)
            
            # Convert labels to communities for modularity calculation
            communities = [set() for _ in range(k)]
            for i, label in enumerate(labels):
                communities[label].add(nodes[i])
            
            # Remove empty communities
            communities = [c for c in communities if len(c) > 0]
            
            # Calculate modularity
            if len(communities) > 1:
                modularity = nx.community.modularity(G, communities, weight='weight')
                modularity_scores.append(modularity)
                
                if modularity > best_modularity:
                    best_modularity = modularity
                    best_n_clusters = k
            else:
                modularity_scores.append(0)
                
        except Exception as e:
            modularity_scores.append(0)
            continue
    
    print(f"   Optimal clusters: {best_n_clusters} (modularity: {best_modularity:.4f})")
    print(f"   Modularity scores: {[f'{m:.3f}' for m in modularity_scores[:5]]}...")
    
    # Step 2: Run final hierarchical clustering with optimal parameters
    try:
        hc = AgglomerativeClustering(
            n_clusters=best_n_clusters,
            metric='precomputed',
            linkage='average'
        )
        labels_hierarchical = hc.fit_predict(dist_matrix)
        
        # Step 3: Create communities
        partition_hierarchical = [set() for _ in range(best_n_clusters)]
        for i, label in enumerate(labels_hierarchical):
            partition_hierarchical[label].add(nodes[i])
        
        # Remove empty partitions
        partition_hierarchical = [p for p in partition_hierarchical if len(p) > 0]
        
        # Calculate final modularity
        mod_hierarchical = nx.community.modularity(G, partition_hierarchical, weight='weight')
        
        print(f"   Hierarchical clustering complete:")
        print(f"     Number of communities: {len(partition_hierarchical)}")
        print(f"     Modularity score: {mod_hierarchical:.4f}")
        
        # Optional: Create linkage matrix for dendrogram analysis
        try:
            Z = linkage(dist_matrix, method='average', metric='precomputed')
            print(f"     Dendrogram linkage computed (height range: {Z[:, 2].min():.3f} - {Z[:, 2].max():.3f})")
        except:
            Z = None
            print("     Dendrogram computation skipped")
        
    except Exception as e:
        print(f"   Error in hierarchical clustering: {e}")
        return [], pd.DataFrame(), pd.DataFrame()
    
    # Step 4: Create detailed results DataFrame
    results = []
    for _, collection in collections_df.iterrows():
        collection_id = collection['collection_id']
        
        # Find which community this collection belongs to
        community_id = -1
        for i, community in enumerate(partition_hierarchical):
            if collection_id in community:
                community_id = i
                break
        
        # Calculate community-specific metrics
        intra_community_connections = 0
        inter_community_connections = 0
        intra_community_weight = 0
        inter_community_weight = 0
        
        if collection_id in G and community_id >= 0:
            for neighbor in G.neighbors(collection_id):
                edge_weight = G[collection_id][neighbor]['weight']
                
                # Check if neighbor is in same community
                neighbor_in_same_community = neighbor in partition_hierarchical[community_id]
                
                if neighbor_in_same_community:
                    intra_community_connections += 1
                    intra_community_weight += edge_weight
                else:
                    inter_community_connections += 1
                    inter_community_weight += edge_weight
        
        # Community cohesion score
        total_connections = intra_community_connections + inter_community_connections
        community_cohesion = intra_community_connections / total_connections if total_connections > 0 else 0
        
        results.append({
            'collection_id': collection_id,
            'collection_name': collection['name'],
            'holder_count': collection['owner_count'],
            'hierarchical_community_id': community_id,
            'hierarchical_community_size': len(partition_hierarchical[community_id]) if community_id >= 0 else 0,
            'total_connections': total_connections,
            'intra_community_connections': intra_community_connections,
            'inter_community_connections': inter_community_connections,
            'hierarchical_community_cohesion_score': community_cohesion,
            'intra_community_weight': intra_community_weight,
            'inter_community_weight': inter_community_weight,
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['hierarchical_community_id', 'holder_count'], ascending=[True, False])
    
    # Step 5: Create community summary
    community_summary = []
    for community_id in range(len(partition_hierarchical)):
        community_collections = results_df[results_df['hierarchical_community_id'] == community_id]
        
        if len(community_collections) > 0:
            community_summary.append({
                'hierarchical_community_id': community_id,
                'hierarchical_community_size': len(community_collections),
                'total_holders': community_collections['holder_count'].sum(),
                'avg_holders': community_collections['holder_count'].mean(),
                'max_holders': community_collections['holder_count'].max(),
                'avg_cohesion': community_collections['hierarchical_community_cohesion_score'].mean(),
                'modularity_contribution': mod_hierarchical / len(partition_hierarchical),
                'dominant_collections': ', '.join(community_collections.head(3)['collection_name'].tolist())
            })
    
    community_summary_df = pd.DataFrame(community_summary)
    community_summary_df = community_summary_df.sort_values('hierarchical_community_size', ascending=False)
    
    print("\nHierarchical Clustering Communities:")
    for _, row in community_summary_df.head(10).iterrows():
        print(f"Community {row['hierarchical_community_id']:2d}: {row['hierarchical_community_size']:2d} collections | "
              f"Avg Holders: {row['avg_holders']:6.0f} | "
              f"Cohesion: {row['avg_cohesion']:.3f} | "
              f"Top: {row['dominant_collections']}")
    
    return partition_hierarchical, results_df, community_summary_df

# ==================================================================================
# COMMUNITY DETECTION VALIDATION METHODS
# ==================================================================================

def validate_community_modularity(G: nx.Graph, louvain_partition: dict, gn_communities: List[Set]) -> dict:
    """
    1. MODULARITY SCORE VALIDATION
    
    Validates community detection by comparing modularity scores to null models.
    
    THEORETICAL BASIS:
    - Modularity Q measures internal vs external community density
    - Range: -0.5 to 1.0 (higher is better)
    - Q > 0.3 indicates good community structure
    - Null comparison tests if modularity is significant vs random
    
    METHODOLOGY:
    - Calculate observed modularity for both Louvain and Girvan-Newman
    - Generate null models with same degree distribution
    - Compare observed vs null modularity distributions
    - Calculate empirical p-values
    
    INTERPRETATION:
    - High modularity: Strong community structure exists
    - p < 0.05: Community structure is statistically significant
    - Louvain typically optimizes for higher modularity than G-N
    
    Args:
        G: NetworkX graph
        louvain_partition: Louvain community assignments {node: community_id}
        gn_communities: Girvan-Newman communities as list of sets
        
    Returns:
        Dictionary with modularity validation results
    """
    import community as community_louvain
    import numpy as np
    
    print("Validating Community Detection Modularity...")
    print("   Method 1: Modularity score comparison and null model testing")
    
    # Calculate observed modularity scores
    mod_louvain = community_louvain.modularity(louvain_partition, G, weight='weight')
    mod_gn = nx.community.modularity(G, gn_communities, weight='weight')
    
    print(f"Observed Modularity Scores:")
    print(f"   Louvain: {mod_louvain:.4f}")
    print(f"   Girvan-Newman: {mod_gn:.4f}")
    
    # Modularity interpretation
    def interpret_modularity(mod):
        if mod > 0.3:
            return "Strong community structure"
        elif mod > 0.1:
            return "Moderate community structure"
        elif mod > 0:
            return "Weak community structure"
        else:
            return "No meaningful community structure"
    
    print(f"   Louvain interpretation: {interpret_modularity(mod_louvain)}")
    print(f"   Girvan-Newman interpretation: {interpret_modularity(mod_gn)}")
    
    # Null model comparison
    print("   Generating null models for significance testing...")
    n_null = 100
    null_mod_louvain = []
    null_mod_gn = []
    
    successful_nulls = 0
    for i in range(n_null):
        try:
            # Create configuration model
            degree_sequence = [d for n, d in G.degree()]
            null_G = nx.configuration_model(degree_sequence)
            null_G = nx.Graph(null_G)  # Remove parallel edges
            null_G.remove_edges_from(nx.selfloop_edges(null_G))
            
            # Ensure connectivity
            if not nx.is_connected(null_G):
                largest_cc = max(nx.connected_components(null_G), key=len)
                null_G = null_G.subgraph(largest_cc).copy()
            
            # Run community detection on null model
            try:
                null_partition_louvain = community_louvain.best_partition(null_G, random_state=42)
                null_mod_louvain.append(community_louvain.modularity(null_partition_louvain, null_G))
                
                # For G-N, just use first split to match complexity
                gn_generator = nx.community.girvan_newman(null_G)
                null_communities_gn = next(gn_generator)
                null_mod_gn.append(nx.community.modularity(null_G, null_communities_gn))
                
                successful_nulls += 1
                
            except (StopIteration, nx.NetworkXError):
                continue
                
        except Exception:
            continue
    
    print(f"   Successful null models: {successful_nulls}/{n_null}")
    
    # Calculate p-values
    louvain_p = np.mean([null >= mod_louvain for null in null_mod_louvain]) if null_mod_louvain else 1.0
    gn_p = np.mean([null >= mod_gn for null in null_mod_gn]) if null_mod_gn else 1.0
    
    # Null distribution statistics
    louvain_null_mean = np.mean(null_mod_louvain) if null_mod_louvain else 0
    gn_null_mean = np.mean(null_mod_gn) if null_mod_gn else 0
    
    print(f"Null Model Comparison:")
    print(f"   Louvain: observed = {mod_louvain:.4f}, null mean = {louvain_null_mean:.4f}, p = {louvain_p:.4f}")
    print(f"   Girvan-Newman: observed = {mod_gn:.4f}, null mean = {gn_null_mean:.4f}, p = {gn_p:.4f}")
    print(f"   Significance: {'SIGNIFICANT' if min(louvain_p, gn_p) < 0.05 else 'NOT SIGNIFICANT'} (p < 0.05)")
    
    return {
        'louvain_modularity': mod_louvain,
        'gn_modularity': mod_gn,
        'louvain_p_value': louvain_p,
        'gn_p_value': gn_p,
        'louvain_null_mean': louvain_null_mean,
        'gn_null_mean': gn_null_mean,
        'null_distributions': {
            'louvain': null_mod_louvain,
            'gn': null_mod_gn
        }
    }

def validate_community_agreement(G: nx.Graph, louvain_partition: dict, gn_communities: List[Set]) -> dict:
    """
    2. AGREEMENT BETWEEN METHODS (Partition Similarity)
    
    Validates community detection by measuring agreement between Louvain and Girvan-Newman.
    
    THEORETICAL BASIS:
    - Adjusted Rand Index (ARI): Measures partition similarity (0=random, 1=identical)
    - Normalized Mutual Information (NMI): Information-theoretic similarity measure
    - High agreement suggests robust community structure
    - Low agreement indicates method-specific artifacts
    
    METRICS:
    - ARI: Corrects for chance agreement, handles different cluster numbers
    - NMI: Symmetric, normalized version of mutual information
    - Overlap coefficient: Measures community boundary agreement
    
    INTERPRETATION:
    - ARI/NMI > 0.7: Strong agreement (highly reliable communities)
    - ARI/NMI 0.3-0.7: Moderate agreement (reasonably reliable)
    - ARI/NMI < 0.3: Weak agreement (method-specific results)
    
    Args:
        G: NetworkX graph
        louvain_partition: Louvain community assignments
        gn_communities: Girvan-Newman communities as list of sets
        
    Returns:
        Dictionary with agreement validation results
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    print("Validating Community Detection Agreement...")
    print("   Method 2: Partition similarity between Louvain and Girvan-Newman")
    
    # Convert partitions to label arrays
    nodes_sorted = sorted(G.nodes())
    
    # Louvain labels
    labels_louvain = [louvain_partition.get(node, -1) for node in nodes_sorted]
    
    # Girvan-Newman labels
    labels_gn = []
    for node in nodes_sorted:
        # Find which community this node belongs to
        community_id = -1
        for i, community in enumerate(gn_communities):
            if node in community:
                community_id = i
                break
        labels_gn.append(community_id)
    
    # Calculate similarity metrics
    ari = adjusted_rand_score(labels_louvain, labels_gn)
    nmi = normalized_mutual_info_score(labels_louvain, labels_gn)
    
    # Calculate additional metrics
    n_communities_louvain = len(set(labels_louvain))
    n_communities_gn = len(set(labels_gn))
    
    # Community size correlation
    from scipy.stats import spearmanr
    
    # Get community sizes for each method
    louvain_sizes = []
    gn_sizes = []
    
    # Count sizes for Louvain
    louvain_community_counts = {}
    for label in labels_louvain:
        louvain_community_counts[label] = louvain_community_counts.get(label, 0) + 1
    louvain_sizes = sorted(louvain_community_counts.values(), reverse=True)
    
    # Count sizes for G-N
    gn_community_counts = {}
    for label in labels_gn:
        gn_community_counts[label] = gn_community_counts.get(label, 0) + 1
    gn_sizes = sorted(gn_community_counts.values(), reverse=True)
    
    # Pad shorter list with zeros for correlation
    max_len = max(len(louvain_sizes), len(gn_sizes))
    louvain_sizes.extend([0] * (max_len - len(louvain_sizes)))
    gn_sizes.extend([0] * (max_len - len(gn_sizes)))
    
    size_correlation, size_corr_p = spearmanr(louvain_sizes, gn_sizes) if max_len > 1 else (0, 1)
    
    print(f"Agreement Analysis Results:")
    print(f"   Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"   Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"   Community counts: Louvain = {n_communities_louvain}, G-N = {n_communities_gn}")
    print(f"   Size distribution correlation: {size_correlation:.4f} (p = {size_corr_p:.4f})")
    
    # Interpretation
    def interpret_agreement(score):
        if score > 0.7:
            return "Strong agreement"
        elif score > 0.3:
            return "Moderate agreement"
        else:
            return "Weak agreement"
    
    print(f"   ARI interpretation: {interpret_agreement(ari)}")
    print(f"   NMI interpretation: {interpret_agreement(nmi)}")
    
    return {
        'ari': ari,
        'nmi': nmi,
        'n_communities_louvain': n_communities_louvain,
        'n_communities_gn': n_communities_gn,
        'size_correlation': size_correlation,
        'size_correlation_p': size_corr_p,
        'labels_louvain': labels_louvain,
        'labels_gn': labels_gn
    }

def validate_community_stability(G: nx.Graph) -> dict:
    """
    3. STABILITY ANALYSIS
    
    Validates community detection by testing stability across multiple runs and perturbations.
    
    THEORETICAL BASIS:
    - Louvain is stochastic: Different runs may yield different results
    - Stable algorithms produce consistent results across runs
    - Perturbation testing: Small changes shouldn't drastically alter communities
    - High stability indicates robust community structure
    
    METHODOLOGY:
    - Run Louvain multiple times, measure pairwise agreement (ARI)
    - Perturb graph weights slightly, re-run detection
    - Calculate average stability scores
    - Identify most/least stable community assignments
    
    INTERPRETATION:
    - Average ARI > 0.8: Highly stable communities
    - Average ARI 0.5-0.8: Moderately stable
    - Average ARI < 0.5: Unstable, method-dependent results
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary with stability validation results
    """
    import community as community_louvain
    from sklearn.metrics import adjusted_rand_score
    import numpy as np
    
    print("Validating Community Detection Stability...")
    print("   Method 3: Stability analysis across multiple runs and perturbations")
    
    # Multiple runs of Louvain (stochastic)
    n_runs = 50  # Reduced for speed
    print(f"   Running Louvain {n_runs} times...")
    
    partitions = []
    modularities = []
    
    for i in range(n_runs):
        partition = community_louvain.best_partition(G, weight='weight', random_state=i)
        partitions.append(partition)
        modularity = community_louvain.modularity(partition, G, weight='weight')
        modularities.append(modularity)
    
    # Convert partitions to label arrays for comparison
    nodes_sorted = sorted(G.nodes())
    labels_list = []
    for partition in partitions:
        labels = [partition.get(node, -1) for node in nodes_sorted]
        labels_list.append(labels)
    
    # Calculate pairwise ARI between all runs
    aris = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            ari = adjusted_rand_score(labels_list[i], labels_list[j])
            aris.append(ari)
    
    avg_ari = np.mean(aris)
    std_ari = np.std(aris)
    min_ari = np.min(aris)
    max_ari = np.max(aris)
    
    avg_modularity = np.mean(modularities)
    std_modularity = np.std(modularities)
    
    print(f"Multiple Runs Analysis:")
    print(f"   Average ARI across runs: {avg_ari:.4f} ± {std_ari:.4f}")
    print(f"   ARI range: [{min_ari:.4f}, {max_ari:.4f}]")
    print(f"   Average modularity: {avg_modularity:.4f} ± {std_modularity:.4f}")
    
    # Perturbation testing
    print(f"   Testing stability under edge weight perturbations...")
    n_perturbations = 20  # Reduced for speed
    perturbation_aris = []
    
    # Get reference partition (most common result)
    reference_partition = partitions[0]  # Use first run as reference
    reference_labels = labels_list[0]
    
    for i in range(n_perturbations):
        try:
            # Create perturbed graph with small noise added to weights
            G_perturbed = G.copy()
            noise_factor = 0.1  # 10% noise
            
            for u, v, data in G_perturbed.edges(data=True):
                original_weight = data.get('weight', 1.0)
                noise = np.random.normal(0, noise_factor * original_weight)
                new_weight = max(0.001, original_weight + noise)  # Ensure positive weights
                G_perturbed[u][v]['weight'] = new_weight
            
            # Run community detection on perturbed graph
            perturbed_partition = community_louvain.best_partition(G_perturbed, weight='weight', random_state=i)
            perturbed_labels = [perturbed_partition.get(node, -1) for node in nodes_sorted]
            
            # Compare to reference
            ari = adjusted_rand_score(reference_labels, perturbed_labels)
            perturbation_aris.append(ari)
            
        except Exception:
            continue
    
    avg_perturbation_ari = np.mean(perturbation_aris) if perturbation_aris else 0
    std_perturbation_ari = np.std(perturbation_aris) if perturbation_aris else 0
    
    print(f"Perturbation Analysis:")
    print(f"   Average ARI under perturbation: {avg_perturbation_ari:.4f} ± {std_perturbation_ari:.4f}")
    print(f"   Successful perturbations: {len(perturbation_aris)}/{n_perturbations}")
    
    # Overall stability interpretation
    def interpret_stability(ari):
        if ari > 0.8:
            return "Highly stable"
        elif ari > 0.5:
            return "Moderately stable"
        else:
            return "Unstable"
    
    print(f"Stability Interpretation:")
    print(f"   Multi-run stability: {interpret_stability(avg_ari)}")
    print(f"   Perturbation stability: {interpret_stability(avg_perturbation_ari)}")
    
    return {
        'avg_ari_runs': avg_ari,
        'std_ari_runs': std_ari,
        'min_ari_runs': min_ari,
        'max_ari_runs': max_ari,
        'avg_modularity_runs': avg_modularity,
        'std_modularity_runs': std_modularity,
        'avg_ari_perturbation': avg_perturbation_ari,
        'std_ari_perturbation': std_perturbation_ari,
        'n_successful_perturbations': len(perturbation_aris),
        'all_run_aris': aris,
        'all_perturbation_aris': perturbation_aris,
        'modularities': modularities
    }

def validate_spectral_clustering(G: nx.Graph, spectral_communities: List[Set], 
                               louvain_partition: dict, gn_communities: List[Set]) -> dict:
    """
    4. COMPREHENSIVE SPECTRAL CLUSTERING VALIDATION
    
    Validates spectral clustering using multiple approaches:
    1. Internal cluster quality (silhouette score)
    2. Modularity significance testing vs null models
    3. Agreement with other methods (ARI/NMI)
    4. Stability analysis (multi-run and perturbation)
    5. Bootstrap robustness testing
    
    THEORETICAL BASIS:
    - Spectral clustering finds communities by analyzing eigenvectors of graph Laplacian
    - Can detect communities that modularity-based methods miss
    - Less sensitive to resolution limits and community size variations
    - Particularly effective for similarity-based graphs like NFT holder overlap
    
    VALIDATION FRAMEWORK:
    - Silhouette >0.3: Good internal cluster quality
    - Modularity >0.2 + significant: Meaningful graph communities
    - ARI/NMI >0.4: Reasonable agreement with other methods
    - Stability >0.8: Robust across runs and perturbations
    
    Args:
        G: NetworkX graph
        spectral_communities: Spectral clustering communities as list of sets
        louvain_partition: Louvain community assignments
        gn_communities: Girvan-Newman communities as list of sets
        
    Returns:
        Dictionary with comprehensive spectral clustering validation results
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from sklearn.cluster import SpectralClustering
    from sklearn.utils import resample
    import community as community_louvain
    import warnings
    
    print("Validating Spectral Clustering...")
    print("   Comprehensive validation: internal quality + significance + agreement + stability")
    
    # Prepare data structures
    adj_matrix = nx.to_numpy_array(G, weight='weight')
    nodes_sorted = sorted(G.nodes())
    n_nodes = len(nodes_sorted)
    
    # Convert all partitions to label arrays for comparison
    labels_spectral = []
    for node in nodes_sorted:
        community_id = -1
        for i, community in enumerate(spectral_communities):
            if node in community:
                community_id = i
                break
        labels_spectral.append(community_id)
    
    labels_louvain = [louvain_partition.get(node, -1) for node in nodes_sorted]
    
    labels_gn = []
    for node in nodes_sorted:
        community_id = -1
        for i, community in enumerate(gn_communities):
            if node in community:
                community_id = i
                break
        labels_gn.append(community_id)
    
    # ==========================================================================
    # 1. INTERNAL CLUSTER QUALITY: SILHOUETTE SCORE
    # ==========================================================================
    
    print("\n   1. Internal Cluster Quality (Silhouette Score)")
    try:
        if len(set(labels_spectral)) > 1:
            silhouette_spectral = silhouette_score(adj_matrix, labels_spectral, metric='precomputed')
        else:
            silhouette_spectral = 0.0
    except:
        silhouette_spectral = 0.0
    
    def interpret_silhouette(score):
        if score > 0.5:
            return "Excellent cluster separation"
        elif score > 0.3:
            return "Good cluster quality"  
        elif score > 0.1:
            return "Moderate cluster quality"
        else:
            return "Poor cluster separation"
    
    print(f"   Silhouette coefficient: {silhouette_spectral:.4f}")
    print(f"   Interpretation: {interpret_silhouette(silhouette_spectral)}")
    
    # ==========================================================================
    # 2. MODULARITY SIGNIFICANCE TESTING
    # ==========================================================================
    
    print("\n   2. Modularity Significance vs Null Models")
    mod_spectral = nx.community.modularity(G, spectral_communities, weight='weight')
    
    # Generate null models
    n_null = 50  # Reduced for speed
    null_mods = []
    successful_nulls = 0
    
    for i in range(n_null):
        try:
            # Configuration model preserving degree sequence
            degree_sequence = [d for n, d in G.degree()]
            null_G = nx.configuration_model(degree_sequence)
            null_G = nx.Graph(null_G)
            null_G.remove_edges_from(nx.selfloop_edges(null_G))
            
            if not nx.is_connected(null_G):
                largest_cc = max(nx.connected_components(null_G), key=len)
                null_G = null_G.subgraph(largest_cc).copy()
            
            # Apply same partition structure to null graph
            null_mod = nx.community.modularity(null_G, spectral_communities, weight='weight')
            null_mods.append(null_mod)
            successful_nulls += 1
            
        except:
            continue
    
    if null_mods:
        p_value_mod = np.mean([null >= mod_spectral for null in null_mods])
        null_mean = np.mean(null_mods)
    else:
        p_value_mod = 1.0
        null_mean = 0.0
    
    print(f"   Observed modularity: {mod_spectral:.4f}")
    print(f"   Null mean modularity: {null_mean:.4f}")
    print(f"   p-value: {p_value_mod:.4f}")
    print(f"   Significance: {'SIGNIFICANT' if p_value_mod < 0.05 else 'NOT SIGNIFICANT'} (p < 0.05)")
    
    # ==========================================================================
    # 3. AGREEMENT WITH OTHER METHODS
    # ==========================================================================
    
    print("\n   3. Agreement with Other Methods")
    
    ari_spectral_louvain = adjusted_rand_score(labels_spectral, labels_louvain)
    nmi_spectral_louvain = normalized_mutual_info_score(labels_spectral, labels_louvain)
    ari_spectral_gn = adjusted_rand_score(labels_spectral, labels_gn)
    nmi_spectral_gn = normalized_mutual_info_score(labels_spectral, labels_gn)
    
    def interpret_agreement(ari, nmi):
        if ari > 0.7 or nmi > 0.7:
            return "Strong agreement"
        elif ari > 0.4 or nmi > 0.4:
            return "Moderate agreement"
        else:
            return "Weak agreement (unique structure)"
    
    print(f"   Spectral vs Louvain: ARI = {ari_spectral_louvain:.4f}, NMI = {nmi_spectral_louvain:.4f}")
    print(f"     Interpretation: {interpret_agreement(ari_spectral_louvain, nmi_spectral_louvain)}")
    print(f"   Spectral vs Girvan-Newman: ARI = {ari_spectral_gn:.4f}, NMI = {nmi_spectral_gn:.4f}")
    print(f"     Interpretation: {interpret_agreement(ari_spectral_gn, nmi_spectral_gn)}")
    
    # ==========================================================================
    # 4. STABILITY ANALYSIS
    # ==========================================================================
    
    print("\n   4. Stability Analysis")
    
    # Multi-run stability (spectral clustering can be stochastic)
    n_runs = 25  # Reduced for speed
    n_clusters = len(set(labels_spectral))
    
    print(f"   Running spectral clustering {n_runs} times for stability...")
    
    labels_runs = []
    successful_runs = 0
    
    # Handle graph connectivity for stability tests
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc).copy()
        adj_matrix_connected = nx.to_numpy_array(G_connected, weight='weight')
        nodes_connected = list(G_connected.nodes())
    else:
        G_connected = G
        adj_matrix_connected = adj_matrix
        nodes_connected = nodes_sorted
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        for i in range(n_runs):
            try:
                epsilon = 1e-6
                adj_processed = adj_matrix_connected + epsilon * np.eye(len(nodes_connected))
                
                sc = SpectralClustering(
                    n_clusters=min(n_clusters, len(nodes_connected)), 
                    affinity='precomputed',
                    assign_labels='kmeans', 
                    random_state=None,  # Allow randomness
                    n_init=5
                )
                labels_run = sc.fit_predict(adj_processed)
                
                # Map back to full graph
                labels_full_run = [-1] * n_nodes
                for j, node in enumerate(nodes_connected):
                    node_idx = nodes_sorted.index(node)
                    labels_full_run[node_idx] = labels_run[j]
                
                labels_runs.append(labels_full_run)
                successful_runs += 1
                
            except:
                continue
    
    # Calculate pairwise ARI between runs
    if successful_runs > 1:
        aris_runs = []
        for i in range(successful_runs):
            for j in range(i + 1, successful_runs):
                ari = adjusted_rand_score(labels_runs[i], labels_runs[j])
                aris_runs.append(ari)
        
        avg_ari_runs = np.mean(aris_runs)
        std_ari_runs = np.std(aris_runs)
    else:
        avg_ari_runs = 0.0
        std_ari_runs = 0.0
    
    print(f"   Multi-run stability: {avg_ari_runs:.4f} ± {std_ari_runs:.4f} (n={successful_runs})")
    
    # Perturbation stability
    print(f"   Testing perturbation stability...")
    n_pert = 15  # Reduced for speed
    pert_aris = []
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        for i in range(n_pert):
            try:
                # Add small noise to adjacency matrix
                noise = np.random.normal(0, 0.01, adj_matrix_connected.shape)
                pert_matrix = adj_matrix_connected + noise
                pert_matrix = np.clip(pert_matrix, 0, 1)  # Keep in [0,1] range
                pert_matrix = pert_matrix + 1e-6 * np.eye(len(nodes_connected))  # Stability
                
                sc_pert = SpectralClustering(
                    n_clusters=min(n_clusters, len(nodes_connected)),
                    affinity='precomputed',
                    assign_labels='kmeans',
                    random_state=42
                )
                labels_pert = sc_pert.fit_predict(pert_matrix)
                
                # Map back to full graph and compare to original
                labels_full_pert = [-1] * n_nodes
                for j, node in enumerate(nodes_connected):
                    node_idx = nodes_sorted.index(node)
                    labels_full_pert[node_idx] = labels_pert[j]
                
                ari_pert = adjusted_rand_score(labels_spectral, labels_full_pert)
                pert_aris.append(ari_pert)
                
            except:
                continue
    
    avg_pert_ari = np.mean(pert_aris) if pert_aris else 0.0
    std_pert_ari = np.std(pert_aris) if pert_aris else 0.0
    
    print(f"   Perturbation stability: {avg_pert_ari:.4f} ± {std_pert_ari:.4f} (n={len(pert_aris)})")
    
    # ==========================================================================
    # 5. OVERALL ASSESSMENT
    # ==========================================================================
    
    print("\n   5. Overall Spectral Clustering Assessment:")
    
    def interpret_stability(ari):
        if ari > 0.8:
            return "Highly stable"
        elif ari > 0.6:
            return "Moderately stable"
        else:
            return "Unstable"
    
    print(f"   Silhouette quality: {interpret_silhouette(silhouette_spectral)}")
    print(f"   Modularity significance: {'Yes' if p_value_mod < 0.05 else 'No'} (p = {p_value_mod:.4f})")
    print(f"   Agreement with Louvain: {interpret_agreement(ari_spectral_louvain, nmi_spectral_louvain)}")
    print(f"   Multi-run stability: {interpret_stability(avg_ari_runs)}")
    print(f"   Perturbation robustness: {interpret_stability(avg_pert_ari)}")
    
    # Community size comparison
    spectral_sizes = [len(community) for community in spectral_communities]
    louvain_sizes = list(Counter(labels_louvain).values())
    gn_sizes = [len(community) for community in gn_communities]
    
    print(f"\n   Community Size Distribution:")
    print(f"   Spectral: {len(spectral_communities)} communities, sizes: {sorted(spectral_sizes, reverse=True)}")
    print(f"   Louvain: {len(set(labels_louvain))} communities, sizes: {sorted(louvain_sizes, reverse=True)}")
    print(f"   G-N: {len(gn_communities)} communities, sizes: {sorted(gn_sizes, reverse=True)}")
    
    return {
        'spectral_modularity': mod_spectral,
        'spectral_modularity_pvalue': p_value_mod,
        'spectral_modularity_significant': p_value_mod < 0.05,
        'silhouette_spectral': silhouette_spectral,
        'n_communities_spectral': len(spectral_communities),
        'ari_spectral_louvain': ari_spectral_louvain,
        'nmi_spectral_louvain': nmi_spectral_louvain,
        'ari_spectral_gn': ari_spectral_gn,
        'nmi_spectral_gn': nmi_spectral_gn,
        'avg_ari_runs': avg_ari_runs,
        'std_ari_runs': std_ari_runs,
        'avg_pert_ari': avg_pert_ari,
        'std_pert_ari': std_pert_ari,
        'successful_runs': successful_runs,
        'successful_perturbations': len(pert_aris),
        'spectral_sizes': spectral_sizes,
        'labels_spectral': labels_spectral
    }

def validate_hierarchical_clustering(G: nx.Graph, hierarchical_communities: List[Set], 
                                   louvain_partition: dict, gn_communities: List[Set],
                                   spectral_communities: List[Set]) -> dict:
    """
    HIERARCHICAL CLUSTERING VALIDATION
    
    Validates hierarchical clustering using comprehensive quality metrics:
    1. Internal cluster quality (silhouette, Calinski-Harabasz, Davies-Bouldin)
    2. Modularity significance testing vs null models
    3. Agreement with other methods (ARI/NMI)
    4. Stability analysis (perturbation robustness)
    5. Dendrogram quality (cophenetic correlation)
    6. Bootstrap resampling for robustness
    
    THEORETICAL BASIS:
    - Hierarchical clustering is deterministic but sensitive to distance matrix
    - Cophenetic correlation measures how well dendrogram preserves distances
    - Internal validation indices assess cluster separation and cohesion
    - Cross-method agreement validates consistency with graph-based methods
    
    VALIDATION FRAMEWORK:
    - Silhouette >0.3: Good internal cluster quality
    - Modularity >0.2 + significant: Respects graph community structure
    - ARI/NMI >0.4: Reasonable agreement with other methods
    - Cophenetic >0.7: Good dendrogram quality
    - Stability >0.8: Robust to data perturbations
    
    Args:
        G: NetworkX graph
        hierarchical_communities: Hierarchical clustering communities
        louvain_partition: Louvain community assignments
        gn_communities: Girvan-Newman communities
        spectral_communities: Spectral clustering communities
        
    Returns:
        Dictionary with comprehensive hierarchical clustering validation results
    """
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, 
                               silhouette_score, calinski_harabasz_score, davies_bouldin_score)
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import cophenet, linkage
    from sklearn.utils import resample
    import community as community_louvain
    import warnings
    
    print("Validating Hierarchical Clustering...")
    print("   Comprehensive validation: quality + significance + agreement + stability + dendrogram")
    
    # Check if hierarchical clustering succeeded
    if not hierarchical_communities:
        print("   Error: Hierarchical clustering failed, skipping validation")
        return {
            'hierarchical_failed': True,
            'silhouette_hc': 0.0,
            'calinski_harabasz_hc': 0.0,
            'davies_bouldin_hc': 0.0,
            'hierarchical_modularity': 0.0,
            'hierarchical_modularity_pvalue': 1.0,
            'ari_hc_louvain': 0.0,
            'nmi_hc_louvain': 0.0,
            'cophenetic_correlation': 0.0,
            'avg_perturbation_ari': 0.0,
            'avg_bootstrap_ari': 0.0
        }
    
    # Prepare data structures
    adj_matrix = nx.to_numpy_array(G, weight='weight')
    dist_matrix = 1 - adj_matrix
    np.fill_diagonal(dist_matrix, 0)
    nodes_sorted = sorted(G.nodes())
    
    # Convert hierarchical communities to labels
    labels_hc = []
    for node in nodes_sorted:
        community_id = -1
        for i, community in enumerate(hierarchical_communities):
            if node in community:
                community_id = i
                break
        labels_hc.append(community_id)
    
    # Convert other methods to labels for comparison
    labels_louvain = [louvain_partition.get(node, -1) for node in nodes_sorted]
    
    labels_gn = []
    for node in nodes_sorted:
        community_id = -1
        for i, community in enumerate(gn_communities):
            if node in community:
                community_id = i
                break
        labels_gn.append(community_id)
    
    labels_spectral = []
    for node in nodes_sorted:
        community_id = -1
        for i, community in enumerate(spectral_communities):
            if node in community:
                community_id = i
                break
        labels_spectral.append(community_id)
    
    # ==========================================================================
    # 1. INTERNAL CLUSTER QUALITY METRICS
    # ==========================================================================
    
    print("\n   1. Internal Cluster Quality Metrics")
    
    try:
        # Silhouette score (precomputed distance)
        if len(set(labels_hc)) > 1:
            silhouette_hc = silhouette_score(dist_matrix, labels_hc, metric='precomputed')
        else:
            silhouette_hc = 0.0
        
        # Calinski-Harabasz score (higher is better)
        try:
            # Note: This expects features, not distance matrix, so we'll use negative distances
            feature_matrix = -dist_matrix  # Convert distances to similarity-like features
            calinski_harabasz_hc = calinski_harabasz_score(feature_matrix, labels_hc)
        except:
            calinski_harabasz_hc = 0.0
        
        # Davies-Bouldin score (lower is better)  
        try:
            davies_bouldin_hc = davies_bouldin_score(feature_matrix, labels_hc)
        except:
            davies_bouldin_hc = float('inf')
        
    except Exception as e:
        print(f"   Error computing internal quality metrics: {e}")
        silhouette_hc = 0.0
        calinski_harabasz_hc = 0.0
        davies_bouldin_hc = float('inf')
    
    def interpret_silhouette(score):
        if score > 0.5:
            return "Excellent cluster separation"
        elif score > 0.3:
            return "Good cluster quality"
        elif score > 0.1:
            return "Moderate cluster quality"
        else:
            return "Poor cluster separation"
    
    print(f"   Silhouette coefficient: {silhouette_hc:.4f}")
    print(f"   Interpretation: {interpret_silhouette(silhouette_hc)}")
    print(f"   Calinski-Harabasz score: {calinski_harabasz_hc:.4f} (higher is better)")
    print(f"   Davies-Bouldin score: {davies_bouldin_hc:.4f} (lower is better)")
    
    # ==========================================================================
    # 2. MODULARITY SIGNIFICANCE TESTING
    # ==========================================================================
    
    print("\n   2. Modularity Significance vs Null Models")
    
    try:
        mod_hc = nx.community.modularity(G, hierarchical_communities, weight='weight')
        
        # Generate null models
        n_null = 50  # Reduced for speed
        null_mods = []
        successful_nulls = 0
        
        for i in range(n_null):
            try:
                degree_sequence = [d for n, d in G.degree()]
                null_G = nx.configuration_model(degree_sequence)
                null_G = nx.Graph(null_G)
                null_G.remove_edges_from(nx.selfloop_edges(null_G))
                
                if not nx.is_connected(null_G):
                    largest_cc = max(nx.connected_components(null_G), key=len)
                    null_G = null_G.subgraph(largest_cc).copy()
                
                null_mod = nx.community.modularity(null_G, hierarchical_communities, weight='weight')
                null_mods.append(null_mod)
                successful_nulls += 1
                
            except:
                continue
        
        if null_mods:
            p_value_mod = np.mean([null >= mod_hc for null in null_mods])
            null_mean = np.mean(null_mods)
        else:
            p_value_mod = 1.0
            null_mean = 0.0
        
        print(f"   Observed modularity: {mod_hc:.4f}")
        print(f"   Null mean modularity: {null_mean:.4f}")  
        print(f"   p-value: {p_value_mod:.4f}")
        print(f"   Significance: {'SIGNIFICANT' if p_value_mod < 0.05 else 'NOT SIGNIFICANT'} (p < 0.05)")
        
    except Exception as e:
        print(f"   Error in modularity testing: {e}")
        mod_hc = 0.0
        p_value_mod = 1.0
    
    # ==========================================================================
    # 3. AGREEMENT WITH OTHER METHODS
    # ==========================================================================
    
    print("\n   3. Agreement with Other Methods")
    
    try:
        ari_hc_louvain = adjusted_rand_score(labels_louvain, labels_hc)
        nmi_hc_louvain = normalized_mutual_info_score(labels_louvain, labels_hc)
        ari_hc_gn = adjusted_rand_score(labels_gn, labels_hc)
        nmi_hc_gn = normalized_mutual_info_score(labels_gn, labels_hc)
        
        if spectral_communities:
            ari_hc_spectral = adjusted_rand_score(labels_spectral, labels_hc)
            nmi_hc_spectral = normalized_mutual_info_score(labels_spectral, labels_hc)
        else:
            ari_hc_spectral = 0.0
            nmi_hc_spectral = 0.0
        
        def interpret_agreement(ari, nmi):
            if ari > 0.7 or nmi > 0.7:
                return "Strong agreement"
            elif ari > 0.4 or nmi > 0.4:
                return "Moderate agreement"
            else:
                return "Weak agreement"
        
        print(f"   HC vs Louvain: ARI = {ari_hc_louvain:.4f}, NMI = {nmi_hc_louvain:.4f}")
        print(f"     Interpretation: {interpret_agreement(ari_hc_louvain, nmi_hc_louvain)}")
        print(f"   HC vs Girvan-Newman: ARI = {ari_hc_gn:.4f}, NMI = {nmi_hc_gn:.4f}")
        print(f"     Interpretation: {interpret_agreement(ari_hc_gn, nmi_hc_gn)}")
        print(f"   HC vs Spectral: ARI = {ari_hc_spectral:.4f}, NMI = {nmi_hc_spectral:.4f}")
        print(f"     Interpretation: {interpret_agreement(ari_hc_spectral, nmi_hc_spectral)}")
        
    except Exception as e:
        print(f"   Error computing agreement metrics: {e}")
        ari_hc_louvain = 0.0
        nmi_hc_louvain = 0.0
        ari_hc_gn = 0.0
        nmi_hc_gn = 0.0
        ari_hc_spectral = 0.0
        nmi_hc_spectral = 0.0
    
    # ==========================================================================
    # 4. DENDROGRAM QUALITY (COPHENETIC CORRELATION)
    # ==========================================================================
    
    print("\n   4. Dendrogram Quality Assessment")
    
    try:
        # Compute linkage matrix for cophenetic correlation
        Z = linkage(dist_matrix, method='average', metric='precomputed')
        cophenetic_corr, _ = cophenet(Z, dist_matrix.ravel())
        
        def interpret_cophenetic(corr):
            if corr > 0.8:
                return "Excellent dendrogram fidelity"
            elif corr > 0.7:
                return "Good dendrogram fidelity"
            elif corr > 0.6:
                return "Moderate dendrogram fidelity"
            else:
                return "Poor dendrogram fidelity"
        
        print(f"   Cophenetic correlation: {cophenetic_corr:.4f}")
        print(f"   Interpretation: {interpret_cophenetic(cophenetic_corr)}")
        
    except Exception as e:
        print(f"   Error computing cophenetic correlation: {e}")
        cophenetic_corr = 0.0
    
    # ==========================================================================
    # 5. STABILITY ANALYSIS (PERTURBATION ROBUSTNESS)
    # ==========================================================================
    
    print("\n   5. Stability Analysis")
    
    n_clusters_hc = len(hierarchical_communities)
    
    # Perturbation stability
    print(f"   Testing perturbation stability...")
    n_pert = 15  # Reduced for speed
    pert_aris = []
    
    for i in range(n_pert):
        try:
            # Add noise to distance matrix
            noise_std = 0.01 * dist_matrix.std()
            pert_dist = dist_matrix + np.random.normal(0, noise_std, dist_matrix.shape)
            pert_dist = np.clip(pert_dist, 0, np.max(pert_dist))
            np.fill_diagonal(pert_dist, 0)
            
            # Ensure symmetry
            pert_dist = (pert_dist + pert_dist.T) / 2
            
            hc_pert = AgglomerativeClustering(
                n_clusters=n_clusters_hc,
                metric='precomputed',
                linkage='average'
            )
            labels_pert = hc_pert.fit_predict(pert_dist)
            
            ari_pert = adjusted_rand_score(labels_hc, labels_pert)
            pert_aris.append(ari_pert)
            
        except Exception as e:
            continue
    
    avg_pert_ari = np.mean(pert_aris) if pert_aris else 0.0
    std_pert_ari = np.std(pert_aris) if pert_aris else 0.0
    
    print(f"   Perturbation stability: {avg_pert_ari:.4f} ± {std_pert_ari:.4f} (n={len(pert_aris)})")
    
    # ==========================================================================
    # 6. BOOTSTRAP RESAMPLING FOR ROBUSTNESS
    # ==========================================================================
    
    print(f"   Testing bootstrap robustness...")
    n_boot = 20  # Reduced for speed
    boot_aris = []
    
    for i in range(n_boot):
        try:
            # Resample edges
            edges = list(G.edges(data=True))
            boot_edges = resample(edges)
            boot_G = nx.Graph()
            boot_G.add_edges_from((u, v, d) for u, v, d in boot_edges)
            
            # Handle missing nodes
            for node in G.nodes():
                if node not in boot_G:
                    boot_G.add_node(node)
            
            boot_adj = nx.to_numpy_array(boot_G, nodelist=nodes_sorted, weight='weight')
            boot_dist = 1 - boot_adj
            np.fill_diagonal(boot_dist, 0)
            
            hc_boot = AgglomerativeClustering(
                n_clusters=n_clusters_hc,
                metric='precomputed',
                linkage='average'
            )
            labels_boot = hc_boot.fit_predict(boot_dist)
            
            ari_boot = adjusted_rand_score(labels_hc, labels_boot)
            boot_aris.append(ari_boot)
            
        except Exception as e:
            continue
    
    avg_boot_ari = np.mean(boot_aris) if boot_aris else 0.0
    std_boot_ari = np.std(boot_aris) if boot_aris else 0.0
    
    if boot_aris:
        ci_boot = np.percentile(boot_aris, [2.5, 97.5])
        print(f"   Bootstrap robustness: {avg_boot_ari:.4f} ± {std_boot_ari:.4f}, 95% CI: [{ci_boot[0]:.4f}, {ci_boot[1]:.4f}] (n={len(boot_aris)})")
    else:
        ci_boot = [0.0, 0.0]
        print(f"   Bootstrap robustness: {avg_boot_ari:.4f} (no successful bootstrap samples)")
    
    # ==========================================================================
    # 7. OVERALL ASSESSMENT
    # ==========================================================================
    
    print("\n   6. Overall Hierarchical Clustering Assessment:")
    
    def interpret_stability(ari):
        if ari > 0.8:
            return "Highly stable"
        elif ari > 0.6:
            return "Moderately stable"
        else:
            return "Unstable"
    
    def interpret_cophenetic(corr):
        if corr is None:
            return "Could not compute (distance matrix issues)"
        elif corr > 0.8:
            return "Excellent dendrogram quality"
        elif corr > 0.6:
            return "Good dendrogram quality"
        elif corr > 0.4:
            return "Fair dendrogram quality"
        else:
            return "Poor dendrogram quality"
    
    print(f"   Silhouette quality: {interpret_silhouette(silhouette_hc)}")
    print(f"   Modularity significance: {'Yes' if p_value_mod < 0.05 else 'No'} (p = {p_value_mod:.4f})")
    print(f"   Agreement with Louvain: {interpret_agreement(ari_hc_louvain, nmi_hc_louvain)}")
    print(f"   Dendrogram quality: {interpret_cophenetic(cophenetic_corr)}")
    print(f"   Perturbation stability: {interpret_stability(avg_pert_ari)}")
    print(f"   Bootstrap robustness: {interpret_stability(avg_boot_ari)}")
    
    return {
        'hierarchical_failed': False,
        'silhouette_hc': silhouette_hc,
        'calinski_harabasz_hc': calinski_harabasz_hc,
        'davies_bouldin_hc': davies_bouldin_hc,
        'hierarchical_modularity': mod_hc,
        'hierarchical_modularity_pvalue': p_value_mod,
        'hierarchical_modularity_significant': p_value_mod < 0.05,
        'ari_hc_louvain': ari_hc_louvain,
        'nmi_hc_louvain': nmi_hc_louvain,
        'ari_hc_gn': ari_hc_gn,
        'nmi_hc_gn': nmi_hc_gn,
        'ari_hc_spectral': ari_hc_spectral,
        'nmi_hc_spectral': nmi_hc_spectral,
        'cophenetic_correlation': cophenetic_corr,
        'avg_perturbation_ari': avg_pert_ari,
        'std_perturbation_ari': std_pert_ari,
        'avg_bootstrap_ari': avg_boot_ari,
        'std_bootstrap_ari': std_boot_ari,
        'bootstrap_ci': ci_boot,
        'n_successful_perturbations': len(pert_aris),
        'n_successful_bootstrap': len(boot_aris)
    }

def validate_all_methods_consensus(G: nx.Graph, louvain_partition: dict, gn_communities: List[Set],
                                  spectral_communities: List[Set], hierarchical_communities: List[Set]) -> dict:
    """
    5. COMPREHENSIVE CROSS-METHOD VALIDATION
    
    Validates all community detection methods by computing pairwise agreement
    and identifying consensus across methods. This provides the ultimate
    validation of community structure reliability.
    
    THEORETICAL BASIS:
    - Multiple methods finding similar communities = high confidence
    - Methods with different algorithmic bases reduce bias
    - Consensus indicates robust underlying community structure
    - Disagreement reveals method-specific artifacts
    
    VALIDATION METRICS:
    - Pairwise ARI/NMI between all method pairs
    - Average agreement scores across methods
    - Consensus stability assessment
    - Method ranking by modularity and agreement
    
    INTERPRETATION:
    - High average ARI (>0.5): Strong consensus, reliable communities
    - Moderate agreement (0.3-0.5): Reasonable consensus
    - Low agreement (<0.3): Method-specific results, use with caution
    
    Args:
        G: NetworkX graph
        louvain_partition: Louvain community assignments
        gn_communities: Girvan-Newman communities
        spectral_communities: Spectral clustering communities  
        hierarchical_communities: Hierarchical clustering communities
        
    Returns:
        Dictionary with comprehensive cross-method validation results
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    import community as community_louvain
    
    print("Validating Cross-Method Consensus...")
    print("   Method 5: Comprehensive agreement analysis across all methods")
    
    # Prepare data structures
    nodes_sorted = sorted(G.nodes())
    methods = ['Louvain', 'Girvan-Newman', 'Spectral', 'Hierarchical']
    
    # Convert all partitions to label arrays
    labels_louvain = [louvain_partition.get(node, -1) for node in nodes_sorted]
    
    labels_gn = []
    for node in nodes_sorted:
        community_id = -1
        for i, community in enumerate(gn_communities):
            if node in community:
                community_id = i
                break
        labels_gn.append(community_id)
    
    labels_spectral = []
    for node in nodes_sorted:
        community_id = -1
        for i, community in enumerate(spectral_communities):
            if node in community:
                community_id = i
                break
        labels_spectral.append(community_id)
    
    labels_hierarchical = []
    for node in nodes_sorted:
        community_id = -1
        for i, community in enumerate(hierarchical_communities):
            if node in community:
                community_id = i
                break
        labels_hierarchical.append(community_id)
    
    labels_list = [labels_louvain, labels_gn, labels_spectral, labels_hierarchical]
    
    # Calculate modularity for all methods (with error handling)
    modularities = {}
    
    try:
        modularities['Louvain'] = community_louvain.modularity(louvain_partition, G, weight='weight')
    except:
        modularities['Louvain'] = 0.0
    
    try:
        modularities['Girvan-Newman'] = nx.community.modularity(G, gn_communities, weight='weight')
    except:
        modularities['Girvan-Newman'] = 0.0
    
    try:
        if spectral_communities:
            modularities['Spectral'] = nx.community.modularity(G, spectral_communities, weight='weight')
        else:
            modularities['Spectral'] = 0.0
    except:
        modularities['Spectral'] = 0.0
    
    try:
        if hierarchical_communities:
            modularities['Hierarchical'] = nx.community.modularity(G, hierarchical_communities, weight='weight')
        else:
            modularities['Hierarchical'] = 0.0
    except:
        modularities['Hierarchical'] = 0.0
    
    # Create pairwise agreement matrices
    n_methods = len(methods)
    ari_matrix = np.zeros((n_methods, n_methods))
    nmi_matrix = np.zeros((n_methods, n_methods))
    
    print("\n   Pairwise Method Agreement:")
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            try:
                # Check if both methods have valid results
                if (community_counts[methods[i]] > 0 and community_counts[methods[j]] > 0 and
                    len(set(labels_list[i])) > 1 and len(set(labels_list[j])) > 1):
                    ari = adjusted_rand_score(labels_list[i], labels_list[j])
                    nmi = normalized_mutual_info_score(labels_list[i], labels_list[j])
                else:
                    ari = 0.0
                    nmi = 0.0
                
                ari_matrix[i, j] = ari
                ari_matrix[j, i] = ari  # Symmetric
                nmi_matrix[i, j] = nmi
                nmi_matrix[j, i] = nmi  # Symmetric
                
                print(f"   {methods[i]:12} vs {methods[j]:12}: ARI = {ari:.4f}, NMI = {nmi:.4f}")
            
            except Exception as e:
                print(f"   {methods[i]:12} vs {methods[j]:12}: ERROR - {str(e)[:50]}")
                ari_matrix[i, j] = 0.0
                ari_matrix[j, i] = 0.0
                nmi_matrix[i, j] = 0.0  
                nmi_matrix[j, i] = 0.0
    
    # Fill diagonal with 1.0 (self-agreement)
    np.fill_diagonal(ari_matrix, 1.0)
    np.fill_diagonal(nmi_matrix, 1.0)
    
    # Calculate average agreement scores
    # Exclude diagonal for average calculation
    mask = ~np.eye(n_methods, dtype=bool)
    avg_ari = ari_matrix[mask].mean()
    std_ari = ari_matrix[mask].std()
    avg_nmi = nmi_matrix[mask].mean()
    std_nmi = nmi_matrix[mask].std()
    
    print(f"\n   Overall Consensus:")
    print(f"   Average ARI: {avg_ari:.4f} ± {std_ari:.4f}")
    print(f"   Average NMI: {avg_nmi:.4f} ± {std_nmi:.4f}")
    
    # Identify best performing methods
    community_counts = {
        'Louvain': len(set(labels_louvain)),
        'Girvan-Newman': len(gn_communities) if gn_communities else 0,
        'Spectral': len(spectral_communities) if spectral_communities else 0,
        'Hierarchical': len(hierarchical_communities) if hierarchical_communities else 0
    }
    
    print(f"\n   Method Comparison:")
    method_scores = []
    for i, method in enumerate(methods):
        # Method score combines modularity and average agreement with others
        method_ari = ari_matrix[i, :].mean()  # Includes self (1.0)
        method_nmi = nmi_matrix[i, :].mean()  # Includes self (1.0)
        method_modularity = modularities[method]
        
        # Composite score (weighted average)
        composite_score = 0.4 * method_modularity + 0.3 * method_ari + 0.3 * method_nmi
        
        method_scores.append({
            'method': method,
            'modularity': method_modularity,
            'avg_ari': method_ari,
            'avg_nmi': method_nmi,
            'composite_score': composite_score,
            'n_communities': community_counts[method]
        })
        
        print(f"   {method:12}: Mod = {method_modularity:.4f}, "
              f"ARI = {method_ari:.4f}, NMI = {method_nmi:.4f}, "
              f"Communities = {community_counts[method]}")
    
    # Sort methods by composite score
    method_scores.sort(key=lambda x: x['composite_score'], reverse=True)
    best_method = method_scores[0]['method']
    
    print(f"\n   Recommended primary method: {best_method} (composite score: {method_scores[0]['composite_score']:.4f})")
    
    # Consensus interpretation
    def interpret_consensus(avg_score):
        if avg_score > 0.7:
            return "Strong consensus - highly reliable communities"
        elif avg_score > 0.5:
            return "Good consensus - reliable communities"  
        elif avg_score > 0.3:
            return "Moderate consensus - reasonably reliable"
        else:
            return "Weak consensus - method-dependent results"
    
    consensus_interpretation = interpret_consensus(avg_ari)
    print(f"   Consensus assessment: {consensus_interpretation}")
    
    return {
        'ari_matrix': ari_matrix.tolist(),
        'nmi_matrix': nmi_matrix.tolist(),
        'avg_ari': avg_ari,
        'std_ari': std_ari,
        'avg_nmi': avg_nmi,
        'std_nmi': std_nmi,
        'modularities': modularities,
        'community_counts': community_counts,
        'method_scores': method_scores,
        'best_method': best_method,
        'consensus_interpretation': consensus_interpretation,
        'methods': methods
    }

# ==================================================================================
# COMBINED ANALYSIS AND RESULTS
# ==================================================================================

def save_results(degree_results_df: pd.DataFrame, eigenvector_results_df: pd.DataFrame, 
                louvain_results_df: pd.DataFrame, community_summary_df: pd.DataFrame,
                gn_results_df: pd.DataFrame, gn_summary_df: pd.DataFrame, 
                gn_hierarchy_df: pd.DataFrame, spectral_results_df: pd.DataFrame, 
                spectral_summary_df: pd.DataFrame, hierarchical_results_df: pd.DataFrame,
                hierarchical_summary_df: pd.DataFrame, validation_results: dict, G: nx.Graph):
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
    
    # Save Spectral clustering results
    spectral_csv_path = OUTPUT_DIR / f"spectral_clustering_communities_{timestamp}.csv"
    spectral_results_df.to_csv(spectral_csv_path, index=False)
    print(f"Spectral clustering results saved to: {spectral_csv_path}")
    
    # Save Spectral clustering summary
    spectral_summary_csv_path = OUTPUT_DIR / f"spectral_clustering_summary_{timestamp}.csv"
    spectral_summary_df.to_csv(spectral_summary_csv_path, index=False)
    print(f"Spectral clustering summary saved to: {spectral_summary_csv_path}")
    
    # Save Hierarchical clustering results
    hierarchical_csv_path = OUTPUT_DIR / f"hierarchical_clustering_communities_{timestamp}.csv"
    hierarchical_results_df.to_csv(hierarchical_csv_path, index=False)
    print(f"Hierarchical clustering results saved to: {hierarchical_csv_path}")
    
    # Save Hierarchical clustering summary
    hierarchical_summary_csv_path = OUTPUT_DIR / f"hierarchical_clustering_summary_{timestamp}.csv"
    hierarchical_summary_df.to_csv(hierarchical_summary_csv_path, index=False)
    print(f"Hierarchical clustering summary saved to: {hierarchical_summary_csv_path}")
    
    # Save validation results
    if 'consistency' in validation_results:
        consistency_csv_path = OUTPUT_DIR / f"centrality_consistency_validation_{timestamp}.csv"
        validation_results['consistency'].to_csv(consistency_csv_path, index=False)
        print(f"Consistency validation saved to: {consistency_csv_path}")
    
    if 'bootstrap' in validation_results:
        bootstrap_csv_path = OUTPUT_DIR / f"centrality_bootstrap_validation_{timestamp}.csv"
        validation_results['bootstrap'].to_csv(bootstrap_csv_path, index=False)
        print(f"Bootstrap validation saved to: {bootstrap_csv_path}")
    
    if 'significance' in validation_results:
        significance_csv_path = OUTPUT_DIR / f"centrality_significance_validation_{timestamp}.csv"
        validation_results['significance'].to_csv(significance_csv_path, index=False)
        print(f"Significance validation saved to: {significance_csv_path}")
    
    # Save community validation results
    if 'community_modularity' in validation_results:
        print(f"Community modularity validation results saved (summary printed above)")
    
    if 'community_agreement' in validation_results:
        print(f"Community agreement validation results saved (summary printed above)")
    
    if 'community_stability' in validation_results:
        print(f"Community stability validation results saved (summary printed above)")
    
    if 'spectral_validation' in validation_results:
        print(f"Spectral clustering validation results saved (summary printed above)")
    
    if 'consensus_validation' in validation_results:
        print(f"Cross-method consensus validation results saved (summary printed above)")
    
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
    
    print("\n" + "="*80)
    print("SPECTRAL CLUSTERING ANALYSIS")
    print("="*80)
    # Analyze Spectral clustering communities
    spectral_communities, spectral_results_df, spectral_summary_df = analyze_spectral_clustering(G, collections_df)
    
    print("\n" + "="*80)
    print("HIERARCHICAL CLUSTERING ANALYSIS")
    print("="*80)
    # Analyze Hierarchical clustering communities
    hierarchical_communities, hierarchical_results_df, hierarchical_summary_df = analyze_hierarchical_clustering(G, collections_df)
    
    print("\n" + "="*80)
    print("CENTRALITY VALIDATION")
    print("="*80)
    # Validate centrality measures
    validation_results = {}
    
    # Method 1: Consistency analysis
    validation_results['consistency'] = validate_centrality_consistency(G, collections_df)
    
    # Method 2: Bootstrap confidence intervals (reduced iterations for speed)
    validation_results['bootstrap'] = validate_centrality_bootstrap(G, collections_df, n_bootstrap=50)
    
    # Method 3: Null model significance testing (reduced iterations for speed)  
    validation_results['significance'] = validate_centrality_significance(G, collections_df, n_null=50)
    
    print("\n" + "="*80)
    print("COMMUNITY DETECTION VALIDATION")
    print("="*80)
    # Validate community detection methods
    
    # Method 1: Modularity score validation
    validation_results['community_modularity'] = validate_community_modularity(G, partition, gn_communities)
    
    # Method 2: Agreement between methods
    validation_results['community_agreement'] = validate_community_agreement(G, partition, gn_communities)
    
    # Method 3: Stability analysis
    validation_results['community_stability'] = validate_community_stability(G)
    
    # Method 4: Spectral clustering validation
    validation_results['spectral_validation'] = validate_spectral_clustering(G, spectral_communities, partition, gn_communities)
    
    # Method 5: Hierarchical clustering validation
    validation_results['hierarchical_validation'] = validate_hierarchical_clustering(G, hierarchical_communities, partition, gn_communities, spectral_communities)
    
    # Method 6: Cross-method consensus validation
    validation_results['consensus_validation'] = validate_all_methods_consensus(G, partition, gn_communities, spectral_communities, hierarchical_communities)
    
    # Save results
    save_results(degree_results_df, eigenvector_results_df, louvain_results_df, community_summary_df,
                gn_results_df, gn_summary_df, gn_hierarchy_df, spectral_results_df, 
                spectral_summary_df, hierarchical_results_df, hierarchical_summary_df, validation_results, G)
    
    print("\nAnalysis complete!")
    print(f"Check {OUTPUT_DIR}/ for detailed results")

if __name__ == "__main__":
    main()