import sys
import os
import multiprocessing
import subprocess
import glob
import pickle
import numpy as np
import networkx as nx
from functools import partial
import library

# Check for cugraph availability
try:
    # We don't import cugraph directly here because we use the networkx backend.
    # However, to fail fast, we can check if it's importable.
    import cugraph
    HAS_CUGRAPH = True
except ImportError:
    HAS_CUGRAPH = False

def get_available_gpus():
    """
    Detects available GPUs using nvidia-smi.
    Returns a list of GPU indices (integers).
    """
    try:
        # Query GPU count
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            encoding="utf-8"
        )
        count = int(result.strip())
        return list(range(count))
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        # Fallback or empty if no nvidia-smi
        # If cugraph is installed, there must be a GPU, but maybe we can't detect it this way.
        # We return [0] as a default if cugraph is present, assuming single GPU.
        # Otherwise empty.
        if HAS_CUGRAPH:
             print("Warning: Could not detect GPUs via nvidia-smi. Assuming 1 GPU (ID 0).")
             return [0]
        return []

def init_worker(gpu_queue):
    """
    Initializer for the worker process.
    Assigns a GPU from the queue to this process.
    """
    try:
        gpu_id = gpu_queue.get_nowait()
    except Exception:
        # Should not happen if queue is populated correctly
        gpu_id = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # print(f"Worker process {os.getpid()} using GPU {gpu_id}")

def process_graph(file_path, cache_dir):
    """
    Worker task to process a single graph file.
    """
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        npy_filename = f"{base_name}_centrality_betweenness.npy"
        npy_path = os.path.join(cache_dir, npy_filename)

        if os.path.exists(npy_path):
            # Resumable: skip if exists
            return

        # Load graph
        with open(file_path, 'rb') as f:
            graph = pickle.load(f)

        # Compute betweenness
        # Note: We rely on networkx >= 2.7 dispatching
        try:
            # We explicitly request the cugraph backend
            scores_dict = nx.betweenness_centrality(graph, backend="cugraph")
        except Exception as e:
            # Fallback or error
            # The user asked to fail fast if cugraph is missing, but here we might be inside a worker.
            # If backend="cugraph" fails, it raises an error.
            print(f"Error computing betweenness for {file_path}: {e}")
            raise e

        # Convert to sorted array
        if graph.number_of_nodes() > 0:
            # We assume nodes are 0..N-1 or at least integers that we can map.
            # library.py implementation assumes:
            # scores_arr = np.array([scores_dict[i] for i in range(graph.number_of_nodes())])
            # This implies the graph nodes are exactly the set {0, ..., N-1}.
            scores_arr = np.array([scores_dict[i] for i in range(graph.number_of_nodes())])
        else:
            scores_arr = np.array([])

        np.save(npy_path, scores_arr)
        print(f"Computed betweenness for {base_name}")

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        # We don't want to kill the pool, but we should report.

def main():
    if not HAS_CUGRAPH:
        print("Error: cugraph module not found. Please install cugraph and ensure dependencies are met.")
        # We allow running without cugraph IF we are just testing logic (handled via mocks in tests),
        # but in production this should fail.
        # For the purpose of the requirement "fail fast", we exit.
        # However, to allow verification in the sandbox (where cugraph is missing),
        # I will check an env var or just warn if I am not in a 'real' run.
        # But the user said "fail fast".
        # I will check if we are in a 'mocked' environment? No.
        # I will exit 1.
        # Exception: If we are running the verification script, we might mock this check or HAS_CUGRAPH.
        sys.exit(1)

    if len(sys.argv) > 1:
        conf_files = sys.argv[1:]
    else:
        conf_files = ['example_config.ini']

    gpus = get_available_gpus()
    num_gpus = len(gpus)
    if num_gpus == 0:
        print("Error: No GPUs detected.")
        sys.exit(1)

    print(f"Detected {num_gpus} GPUs: {gpus}")

    for conf_file in conf_files:
        print(f"Processing config: {conf_file}")
        try:
            config = library.get_config_from_ini(conf_file)
        except Exception as e:
            print(f"Error loading config {conf_file}: {e}")
            continue

        # Discover files
        # We reuse library.get_cached_files logic
        files = library.get_cached_files(config)

        if not files:
            print(f"No cached graph files found for {conf_file}")
            continue

        print(f"Found {len(files)} graph files.")

        # Prepare pool
        manager = multiprocessing.Manager()
        gpu_queue = manager.Queue()
        # Populate queue with GPU IDs - round robin if more jobs?
        # No, the pool size is num_gpus. Each worker takes one ID.
        # We need exactly num_gpus tokens.
        for gpu_id in gpus:
            gpu_queue.put(gpu_id)

        # Use 'spawn' context for CUDA compatibility
        ctx = multiprocessing.get_context('spawn')

        # Function wrapper to include cache_dir
        worker_func = partial(process_graph, cache_dir=config.cache_dir)

        # We need to map the list of files
        # If we have many files, we want the workers to pick them up.
        # The pool initializer runs once per process start.
        with ctx.Pool(processes=num_gpus, initializer=init_worker, initargs=(gpu_queue,)) as pool:
            pool.map(worker_func, files)

if __name__ == "__main__":
    main()
