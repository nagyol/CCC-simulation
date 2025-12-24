
import time
import networkx as nx
import numpy as np
import library
import main

def benchmark_overlap(n=10000):
    print(f"\n--- Benchmarking Overlap Calculation (N={n}) ---")
    baseline = np.random.random(n).tolist()
    other = np.random.random(n).tolist()

    start = time.time()
    library.compare_centrality(baseline, other)
    end = time.time()
    print(f"Overlap calculation time: {end - start:.4f}s")

def benchmark_centralities(n=2000):
    print(f"\n--- Benchmarking Centrality Calculations (N={n}) ---")
    print("Generating graph...")
    # Using Erdos-Renyi for consistent benchmarking
    G = nx.fast_gnp_random_graph(n, 0.01)

    centralities = [
        ("PageRank", library.get_ranking_pagerank),
        ("Betweenness", library.get_ranking_betweenness),
        ("Closeness", library.get_ranking_closeness),
        ("Eigenvector", library.get_ranking_eigenvector),
    ]

    for name, func in centralities:
        print(f"Running {name}...")
        start = time.time()
        func(G)
        end = time.time()
        print(f"  {name} time: {end - start:.4f}s")

def dummy_task():
    return 1

def benchmark_parallel(runs=50):
    print(f"\n--- Benchmarking Parallel Execution Overhead (Runs={runs}) ---")

    start = time.time()
    library.run_in_parallel(runs, dummy_task)
    end = time.time()
    print(f"Parallel execution time: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark_overlap()
    benchmark_centralities()
    benchmark_parallel()
