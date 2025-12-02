
import time
import networkx as nx
import numpy as np
import rustworkx as rx
import pandas as pd
import os
import sys

try:
    import grape
    HAS_GRAPE = True
except ImportError:
    HAS_GRAPE = False
try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False

import warnings
warnings.filterwarnings("ignore")

def benchmark_all(n=2000, runs=1):
    print(f"\n==================================================", flush=True)
    print(f"BENCHMARK: N={n}, Graph=Erdos-Renyi (p=0.01)", flush=True)
    print(f"==================================================", flush=True)

    # Generate NetworkX graph first (Baseline)
    print("Generating NetworkX graph...", flush=True)
    G_nx = nx.fast_gnp_random_graph(n, 0.01, seed=42)
    if not nx.is_connected(G_nx):
        G_nx = G_nx.subgraph(max(nx.connected_components(G_nx), key=len)).copy()
        print(f"  Using largest component, N={G_nx.number_of_nodes()}", flush=True)

    # Convert to igraph
    G_ig = None
    if HAS_IGRAPH:
        print("Converting to igraph...", flush=True)
        G_ig = ig.Graph.from_networkx(G_nx)

    # Convert to rustworkx
    print("Converting to rustworkx...", flush=True)
    G_rx = rx.networkx_converter(G_nx)

    # Convert to GRAPE
    G_grape = None
    if HAS_GRAPE:
        print("Converting to GRAPE...", flush=True)
        df = nx.to_pandas_edgelist(G_nx)
        df.to_csv("temp_edges.csv", index=False)
        try:
            G_grape = grape.Graph.from_csv(
                edge_path="temp_edges.csv",
                edge_list_separator=",",
                edge_list_header=True,
                sources_column="source",
                destinations_column="target",
                directed=False,
                name="benchmark_graph",
                verbose=False
            )
        except Exception as e:
            print(f"GRAPE conversion failed: {e}", flush=True)
        finally:
            if os.path.exists("temp_edges.csv"):
                os.remove("temp_edges.csv")

    metrics = ["PageRank", "Betweenness", "Closeness", "Eigenvector", "Degree"]

    # --- NetworkX ---
    if n <= 2000:
        print("\n--- NetworkX ---", flush=True)
        for metric in metrics:
            start = time.time()
            try:
                if metric == "PageRank":
                    nx.pagerank(G_nx)
                elif metric == "Betweenness":
                    nx.betweenness_centrality(G_nx)
                elif metric == "Closeness":
                    nx.closeness_centrality(G_nx)
                elif metric == "Eigenvector":
                    nx.eigenvector_centrality(G_nx)
                elif metric == "Degree":
                    nx.degree_centrality(G_nx)
                dur = time.time() - start
                print(f"{metric}: {dur:.4f}s", flush=True)
            except Exception as e:
                print(f"{metric}: Failed ({e})", flush=True)
    else:
        print("\n--- NetworkX (Skipped for large N) ---", flush=True)

    # --- igraph ---
    if HAS_IGRAPH:
        print("\n--- igraph ---", flush=True)
        for metric in metrics:
            start = time.time()
            try:
                if metric == "PageRank":
                    G_ig.pagerank()
                elif metric == "Betweenness":
                    G_ig.betweenness()
                elif metric == "Closeness":
                    G_ig.closeness()
                elif metric == "Eigenvector":
                    G_ig.eigenvector_centrality(scale=False)
                elif metric == "Degree":
                    G_ig.degree()
                dur = time.time() - start
                print(f"{metric}: {dur:.4f}s", flush=True)
            except Exception as e:
                print(f"{metric}: Failed ({e})", flush=True)

    # --- rustworkx ---
    print("\n--- rustworkx ---", flush=True)
    for metric in metrics:
        start = time.time()
        try:
            if metric == "PageRank":
                rx.pagerank(G_rx)
            elif metric == "Betweenness":
                rx.betweenness_centrality(G_rx)
            elif metric == "Closeness":
                rx.closeness_centrality(G_rx)
            elif metric == "Eigenvector":
                rx.eigenvector_centrality(G_rx)
            elif metric == "Degree":
                rx.degree_centrality(G_rx)
            dur = time.time() - start
            print(f"{metric}: {dur:.4f}s", flush=True)
        except Exception as e:
            print(f"{metric}: Failed ({e})", flush=True)

    # --- GRAPE ---
    if HAS_GRAPE and G_grape:
        print("\n--- GRAPE ---", flush=True)
        for metric in metrics:
            start = time.time()
            try:
                if metric == "PageRank":
                    # Try to find pagerank, if not, skip
                     if hasattr(G_grape, "get_pagerank"):
                         G_grape.get_pagerank()
                     else:
                         print("PageRank: Not Found", flush=True)
                         continue
                elif metric == "Betweenness":
                     if hasattr(G_grape, "get_betweenness_centrality"):
                         G_grape.get_betweenness_centrality()
                     elif hasattr(G_grape, "get_approximated_betweenness_centrality"):
                         G_grape.get_approximated_betweenness_centrality()
                         print("(approximated)", end=" ", flush=True)
                     else:
                         print("Betweenness: Not Found", flush=True)
                         continue
                elif metric == "Closeness":
                    G_grape.get_closeness_centrality()
                elif metric == "Eigenvector":
                     if hasattr(G_grape, "get_eigenvector_centrality"):
                         G_grape.get_eigenvector_centrality()
                     elif hasattr(G_grape, "get_weighted_eigenvector_centrality"):
                         G_grape.get_weighted_eigenvector_centrality()
                     else:
                         print("Eigenvector: Not Found", flush=True)
                         continue
                elif metric == "Degree":
                    G_grape.get_degree_centrality()
                dur = time.time() - start
                print(f"{metric}: {dur:.4f}s", flush=True)
            except Exception as e:
                print(f"{metric}: Failed ({e})", flush=True)

if __name__ == "__main__":
    benchmark_all(n=2000)
    benchmark_all(n=10000)
