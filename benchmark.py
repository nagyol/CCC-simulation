
import time
import numpy as np
import main
import library
import typing

def benchmark_compare_centrality(n=5000):
    print(f"Benchmarking compare_centrality with N={n}...")
    # Generate two random rankings (permutations of 0..n-1)
    # The original code expects lists of values, which are then sorted to get orderings.
    # main.compare_centrality takes: (baseline: typing.List, ranking_other: typing.List)
    # These are lists of centrality values for each node index.

    baseline_values = np.random.random(n).tolist()
    other_values = np.random.random(n).tolist()

    start_time = time.time()
    main.compare_centrality(baseline_values, other_values)
    end_time = time.time()

    print(f"compare_centrality took {end_time - start_time:.4f} seconds")

def benchmark_parallel_overhead(runs=100):
    print(f"Benchmarking parallel overhead with runs={runs}...")

    def dummy_task():
        return {'overlap': []}

    start_time = time.time()
    # library.run_in_parallel expects a function that returns a result
    # It passes it to apply_async.
    library.run_in_parallel(runs, dummy_task)
    end_time = time.time()

    print(f"run_in_parallel took {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark_compare_centrality(n=2000)
    benchmark_compare_centrality(n=5000)
    # benchmark_parallel_overhead(runs=100) # This might require more setup as run_in_parallel is in library
