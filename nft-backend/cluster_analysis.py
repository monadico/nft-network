import sqlite3
import pandas as pd
import networkx as nx
import numpy as np
import json
import itertools
import os
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans

# Configuration for density thresholds
DENSITY_THRESHOLDS = {
    'low': 5,
    'medium': 20, 
    'high': 50
}

# Configuration for Jaccard similarity thresholds
JACCARD_THRESHOLDS = {
    'low': 0.01,
    'medium': 0.05,
    'high': 0.1
}

# Output directory for JSON files
OUTPUT_DIR = Path('json_clusters')

def ensure_output_directory():
    """Ensure the output directory exists"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")

def load_data_from_sqlite():
    """Load holder data from SQLite database"""
    print("📊 Loading data from SQLite...")
    
    conn = sqlite3.connect('data/holders.sqlite')
    
    # Load collection_holders data
    df = pd.read_sql("""
        SELECT 
            ch.collection_id,
            c.name as collection_name,
            ch.holder_address,
            ch.token_count
        FROM collection_holders ch
        JOIN collections c ON ch.collection_id = c.collection_id
    """, conn)
    
    conn.close()
    
    print(f"✅ Loaded {len(df)} holder records across {df['collection_id'].nunique()} collections")
    return df

def compute_overlap_matrix(df):
    """Compute pairwise overlaps between collections"""
    print("🔗 Computing overlap matrix...")
    
    # Get unique collections
    collections = df['collection_id'].unique()
    collection_names = {c: df[df['collection_id'] == c]['collection_name'].iloc[0] for c in collections}
    
    # Build sets of holders per collection
    holders = {c: set(df[df['collection_id'] == c]['holder_address']) for c in collections}
    
    # Compute pairwise overlaps
    edges = []
    for c1, c2 in itertools.combinations(collections, 2):
        shared = holders[c1] & holders[c2]
        weight = len(shared)
        if weight > 0:
            edges.append({
                "source": c1,
                "target": c2,
                "weight": weight,
                "source_name": collection_names[c1],
                "target_name": collection_names[c2]
            })
    
    print(f"✅ Found {len(edges)} connections between collections")
    return collections, collection_names, holders, edges

def compute_jaccard_similarity(edges, holders):
    """Compute Jaccard similarity for each edge"""
    print("🧮 Computing Jaccard similarity...")
    
    jaccard_edges = []
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        
        # Get holder sets
        holders_a = holders[source]
        holders_b = holders[target]
        
        # Compute Jaccard similarity
        intersection = len(holders_a & holders_b)  # |A ∩ B|
        union = len(holders_a | holders_b)        # |A ∪ B|
        
        if union > 0:
            jaccard = intersection / union
            jaccard_edge = edge.copy()
            jaccard_edge["jaccard"] = jaccard
            jaccard_edges.append(jaccard_edge)
    
    print(f"✅ Computed Jaccard similarity for {len(jaccard_edges)} edges")
    return jaccard_edges

def build_graph(collections, collection_names, holders, edges):
    """Build NetworkX graph from overlap data"""
    print("🕸️ Building network graph...")
    
    G = nx.Graph()
    
    # Add nodes
    for c in collections:
        G.add_node(c, 
                   name=collection_names[c],
                   size=len(holders[c]))
    
    # Add edges
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e["weight"])
    
    print(f"✅ Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def detect_communities(G):
    """Detect communities using Louvain algorithm"""
    print("🏘️ Detecting communities...")
    
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, weight="weight")
        nx.set_node_attributes(G, partition, "community")
        print(f"✅ Louvain communities detected: {len(set(partition.values()))} communities")
    except ImportError:
        print("⚠️ python-louvain not installed, skipping community detection")
        # Set default community
        for node in G.nodes():
            G.nodes[node]["community"] = 0
    
    return G

def hierarchical_clustering(collections, edges):
    """Perform hierarchical clustering"""
    print("🌳 Performing hierarchical clustering...")
    
    # Build overlap matrix
    matrix = np.zeros((len(collections), len(collections)))
    collection_list = list(collections)
    
    for e in edges:
        i = collection_list.index(e["source"])
        j = collection_list.index(e["target"])
        matrix[i, j] = matrix[j, i] = e["weight"]
    
    # Hierarchical clustering
    try:
        Z = linkage(matrix, method="ward")
        # Try to find optimal number of clusters
        n_clusters = min(10, len(collections) // 2)  # Adaptive clustering
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        
        # Map labels back to collections
        for i, c in enumerate(collection_list):
            G.nodes[c]["hierarchical"] = int(labels[i] - 1)  # 0-indexed
        
        print(f"✅ Hierarchical clustering: {n_clusters} clusters")
    except Exception as e:
        print(f"⚠️ Hierarchical clustering failed: {e}")
        for node in G.nodes():
            G.nodes[node]["hierarchical"] = 0
    
    return G

def kmeans_clustering(collections, edges):
    """Perform K-means clustering"""
    print("🎯 Performing K-means clustering...")
    
    # Build overlap matrix
    matrix = np.zeros((len(collections), len(collections)))
    collection_list = list(collections)
    
    for e in edges:
        i = collection_list.index(e["source"])
        j = collection_list.index(e["target"])
        matrix[i, j] = matrix[j, i] = e["weight"]
    
    # K-means clustering
    try:
        n_clusters = min(8, len(collections) // 3)  # Adaptive clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km_labels = kmeans.fit_predict(matrix)
        
        for i, c in enumerate(collection_list):
            G.nodes[c]["kmeans"] = int(km_labels[i])
        
        print(f"✅ K-means clustering: {n_clusters} clusters")
    except Exception as e:
        print(f"⚠️ K-means clustering failed: {e}")
        for node in G.nodes():
            G.nodes[node]["kmeans"] = 0
    
    return G

def compute_centrality_metrics(G):
    """Compute centrality metrics for each node"""
    print("📈 Computing centrality metrics...")
    
    # Degree centrality
    deg_centrality = nx.degree_centrality(G)
    nx.set_node_attributes(G, deg_centrality, "degree_centrality")
    
    # Betweenness centrality
    try:
        bet_centrality = nx.betweenness_centrality(G, weight="weight")
        nx.set_node_attributes(G, bet_centrality, "betweenness_centrality")
    except:
        print("⚠️ Betweenness centrality failed, using degree centrality")
        bet_centrality = deg_centrality
        nx.set_node_attributes(G, bet_centrality, "betweenness_centrality")
    
    # Eigenvector centrality
    try:
        eig_centrality = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
        nx.set_node_attributes(G, eig_centrality, "eigenvector_centrality")
    except:
        print("⚠️ Eigenvector centrality failed, using degree centrality")
        eig_centrality = deg_centrality
        nx.set_node_attributes(G, eig_centrality, "eigenvector_centrality")
    
    print("✅ Centrality metrics computed")
    return G

def filter_edges_by_threshold(edges, threshold, threshold_type="weight"):
    """Filter edges based on threshold"""
    if threshold_type == "weight":
        filtered_edges = [e for e in edges if e["weight"] >= threshold]
    elif threshold_type == "jaccard":
        filtered_edges = [e for e in edges if e.get("jaccard", 0) >= threshold]
    else:
        filtered_edges = edges
    
    return filtered_edges

def export_density_based_jsons(G, edges, holders):
    """Export JSONs based on density thresholds"""
    print("\n📊 Exporting density-based JSONs...")
    
    for density_level, threshold in DENSITY_THRESHOLDS.items():
        print(f"  🔸 Processing {density_level} density (threshold: {threshold})...")
        
        # Filter edges by weight threshold
        filtered_edges = filter_edges_by_threshold(edges, threshold, "weight")
        
        # Get unique nodes from filtered edges
        node_ids = set()
        for edge in filtered_edges:
            node_ids.add(edge["source"])
            node_ids.add(edge["target"])
        
        # Create data structure with metadata
        data = {
            "metadata": {
                "edge_count": len(filtered_edges),
                "node_count": len(node_ids),
                "threshold": threshold,
                "normalized": False,
                "description": f"Raw holder overlaps with minimum {threshold} shared holders"
            },
            "nodes": [
                {
                    "id": n,
                    "name": G.nodes[n].get("name", n),
                    "size": G.nodes[n].get("size", 0),
                    "community": G.nodes[n].get("community", 0),
                    "hierarchical": G.nodes[n].get("hierarchical", 0),
                    "kmeans": G.nodes[n].get("kmeans", 0),
                    "degree_centrality": round(G.nodes[n].get("degree_centrality", 0), 6),
                    "betweenness_centrality": round(G.nodes[n].get("betweenness_centrality", 0), 6),
                    "eigenvector_centrality": round(G.nodes[n].get("eigenvector_centrality", 0), 6)
                }
                for n in node_ids
            ],
            "edges": [
                {
                    "source": e["source"],
                    "target": e["target"],
                    "weight": e["weight"],
                    "source_name": e["source_name"],
                    "target_name": e["target_name"]
                }
                for e in filtered_edges
            ]
        }
        
        # Export to file
        filename = f"{density_level}_density.json"
        with open(OUTPUT_DIR / filename, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"    ✅ {filename} → {len(node_ids)} nodes, {len(filtered_edges)} edges, threshold {threshold}, normalized False")

def export_jaccard_based_jsons(G, jaccard_edges, holders):
    """Export JSONs based on Jaccard similarity thresholds"""
    print("\n🧮 Exporting Jaccard similarity JSONs...")
    
    for jaccard_level, threshold in JACCARD_THRESHOLDS.items():
        print(f"  🔸 Processing {jaccard_level} Jaccard similarity (threshold: {threshold})...")
        
        # Filter edges by Jaccard threshold
        filtered_edges = filter_edges_by_threshold(jaccard_edges, threshold, "jaccard")
        
        # Get unique nodes from filtered edges
        node_ids = set()
        for edge in filtered_edges:
            node_ids.add(edge["source"])
            node_ids.add(edge["target"])
        
        # Create data structure with metadata
        data = {
            "metadata": {
                "edge_count": len(filtered_edges),
                "node_count": len(node_ids),
                "threshold": threshold,
                "normalized": True,
                "description": f"Jaccard similarity with minimum {threshold} similarity score"
            },
            "nodes": [
                {
                    "id": n,
                    "name": G.nodes[n].get("name", n),
                    "size": G.nodes[n].get("size", 0),
                    "community": G.nodes[n].get("community", 0),
                    "hierarchical": G.nodes[n].get("hierarchical", 0),
                    "kmeans": G.nodes[n].get("kmeans", 0),
                    "degree_centrality": round(G.nodes[n].get("degree_centrality", 0), 6),
                    "betweenness_centrality": round(G.nodes[n].get("betweenness_centrality", 0), 6),
                    "eigenvector_centrality": round(G.nodes[n].get("eigenvector_centrality", 0), 6)
                }
                for n in node_ids
            ],
            "edges": [
                {
                    "source": e["source"],
                    "target": e["target"],
                    "weight": round(e["jaccard"], 6),  # Use Jaccard value as weight
                    "source_name": e["source_name"],
                    "target_name": e["target_name"]
                }
                for e in filtered_edges
            ]
        }
        
        # Export to file
        filename = f"{jaccard_level}_density_normalized.json"
        with open(OUTPUT_DIR / filename, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"    ✅ {filename} → {len(node_ids)} nodes, {len(filtered_edges)} edges, threshold {threshold}, normalized True")

def export_to_json(G, edges):
    """Export graph data to JSON for frontend (original function)"""
    print("💾 Exporting original JSON...")
    
    data = {
        "metadata": {
            "edge_count": len(edges),
            "node_count": G.number_of_nodes(),
            "threshold": 0,
            "normalized": False,
            "description": "Complete network with all edges and nodes (no filtering applied)"
        },
        "nodes": [
            {
                "id": n,
                "name": G.nodes[n].get("name", n),
                "size": G.nodes[n].get("size", 0),
                "community": G.nodes[n].get("community", 0),
                "hierarchical": G.nodes[n].get("hierarchical", 0),
                "kmeans": G.nodes[n].get("kmeans", 0),
                "degree_centrality": round(G.nodes[n].get("degree_centrality", 0), 6),
                "betweenness_centrality": round(G.nodes[n].get("betweenness_centrality", 0), 6),
                "eigenvector_centrality": round(G.nodes[n].get("eigenvector_centrality", 0), 6)
            }
            for n in G.nodes()
        ],
        "edges": [
            {
                "source": e["source"],
                "target": e["target"],
                "weight": e["weight"],
                "source_name": e["source_name"],
                "target_name": e["target_name"]
            }
            for e in edges
        ]
    }
    
    # Save to file
    output_file = "nft_network.json"
    with open(OUTPUT_DIR / output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ {output_file} → {G.number_of_nodes()} nodes, {len(edges)} edges, threshold 0, normalized False")
    
    return data

def main():
    """Main pipeline execution"""
    print("🚀 Starting Enhanced NFT Holder Clustering Analysis")
    print("=" * 60)
    
    # Ensure output directory exists
    ensure_output_directory()
    
    # Step 1: Load data
    df = load_data_from_sqlite()
    
    # Step 2: Compute overlaps
    collections, collection_names, holders, edges = compute_overlap_matrix(df)
    
    # Step 3: Compute Jaccard similarity
    jaccard_edges = compute_jaccard_similarity(edges, holders)
    
    # Step 4: Build graph
    global G  # Make G global for other functions
    G = build_graph(collections, collection_names, holders, edges)
    
    # Step 5: Community detection
    G = detect_communities(G)
    
    # Step 6: Hierarchical clustering
    G = hierarchical_clustering(collections, edges)
    
    # Step 7: K-means clustering
    G = kmeans_clustering(collections, edges)
    
    # Step 8: Centrality metrics
    G = compute_centrality_metrics(G)
    
    # Step 9: Export original JSON
    data = export_to_json(G, edges)
    
    # Step 10: Export density-based JSONs
    export_density_based_jsons(G, edges, holders)
    
    # Step 11: Export Jaccard-based JSONs
    export_jaccard_based_jsons(G, jaccard_edges, holders)
    
    print("\n🎉 Enhanced analysis complete!")
    print("📁 Generated 7 JSON files:")
    print("   • nft_network.json (original)")
    print("   • low_density.json")
    print("   • medium_density.json") 
    print("   • high_density.json")
    print("   • low_density_normalized.json")
    print("   • medium_density_normalized.json")
    print("   • high_density_normalized.json")
    print("🖥️ Ready for frontend visualization!")

if __name__ == "__main__":
    main()
