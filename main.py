import typing
from functools import partial
import pickle
import multiprocessing
import numpy as np

import library

# Note: compare_centrality and related functions have been moved to library.py
# We can import them if needed, or rely on library calls.

def plot_overlap(results: typing.List, baseline: typing.AnyStr, centrality: typing.AnyStr, zoom: bool = True,
                 note: typing.AnyStr = None) -> None:
    library.plot_general(results, baseline=baseline, centrality=centrality, header="", zoom=zoom,
                         parabola=True, rescale=True, note=note)
    return


def get_ranking(centrality: typing.AnyStr):
    # This function is also in library.py now, but main.py might use it.
    # However, if we use the cache workflow, we don't call this directly from main usually.
    # But for non-cache workflow, we still need it.
    # Let's delegate to library.
    return library.get_ranking(centrality)


def simulate_one_scenario(in_centrality1: typing.AnyStr, in_centrality2: typing.AnyStr, in_configuration: typing.NamedTuple, run_index: int = None, cached_files: typing.List[str] = None):
    # If using cache, we use the .npy files corresponding to the graph
    if in_configuration.cache and cached_files is not None and run_index is not None and run_index < len(cached_files):
        file_path = cached_files[run_index]
        try:
            # We don't load the graph anymore! Speed up!
            # We just need the centrality arrays.
            # Filename convention: {base}_centrality_{name}.npy

            import os
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            npy1 = os.path.join(in_configuration.cache_dir, f"{base_name}_centrality_{in_centrality1}.npy")
            npy2 = os.path.join(in_configuration.cache_dir, f"{base_name}_centrality_{in_centrality2}.npy")

            # Load arrays
            # We assume these exist because process_graph ensured it.
            c1_scores = np.load(npy1)
            c2_scores = np.load(npy2)

            # compare_centrality expects list or dict. Array is list-like.
            # It accepts ndarray if we verify library.py handles it.
            # library.get_normalized_total_orderings handles array input.

            return library.compare_centrality(c1_scores, c2_scores)

        except Exception as e:
            print(f"Error loading cached centralities for {file_path}: {e}. Falling back to generation.")
            # Fallback to generation below
            pass

    # Default behavior (no cache or fallback)
    graph = in_configuration.generator()
    # When generating fresh, we get dicts.
    return library.compare_centrality(get_ranking(in_centrality1)(graph), get_ranking(in_centrality2)(graph))


def run_comparing_sim(runs: int, centrality1: typing.AnyStr, centrality2: typing.AnyStr, configuration: typing.NamedTuple,
                      note: typing.AnyStr = None, parallel: bool = True, cached_files: typing.List[str] = None) -> None:

    print(f'Running: {centrality1} vs. {centrality2}, conffile: {configuration.name}')

    # We pass run_index to simulate_one_scenario
    # If parallel, we need to pass arguments correctly.
    # partial can fix some args, but we need varying run_index.

    tasks = []
    for i in range(runs):
        tasks.append(partial(simulate_one_scenario, centrality1, centrality2, configuration, run_index=i, cached_files=cached_files))

    if not parallel:
        sim_results = [task() for task in tasks]
    else:
        # library.run_in_parallel expects a function that takes 0 args?
        # library.run_in_parallel signature: (runs, fn). It runs fn() 'runs' times.
        # But we need specific arguments for each run (the index).
        # library.run_in_parallel is implemented as: [pool.apply_async(fn) for _ in range(runs)].
        # It calls the SAME function 'runs' times.
        # This works for random generation (fresh seed/state).
        # But for caching, we need 'i'.

        # We need to modify how we call parallel execution or use a different helper.
        # Since we are modifying main.py, we can just use multiprocessing directly here or update library.run_in_parallel.
        # Updating library.run_in_parallel to accept a list of functions would be better.
        # Or just do it here.

        with multiprocessing.Pool() as pool:
            # tasks is a list of partials.
            # pool.map(lambda f: f(), tasks) ?
            # tasks are not picklable if they contain local functions? Partial is picklable.

            # Simpler: map an index.
            worker = partial(simulate_one_scenario, centrality1, centrality2, configuration, cached_files=cached_files)
            # Worker takes (run_index=...) as kwarg? No, map passes positional.
            # simulate_one_scenario(c1, c2, conf, index, files)

            # We need a wrapper to adapt arguments for map.
            # Or use starmap.

            # Let's define a worker helper (top level for picklability)
            # But simulate_one_scenario is top level.
            # We can use starmap.
            pass

            args = [(centrality1, centrality2, configuration, i, cached_files) for i in range(runs)]
            sim_results = pool.starmap(simulate_one_scenario, args)

    results = {'overlap': [res['overlap'] for res in sim_results]}
    plot_overlap(results['overlap'], baseline=centrality1, centrality=centrality2, zoom=False, note=note)


import sys

def main():

    if len(sys.argv) > 1:
        all_conf_files = sys.argv[1:]
    else:
        all_conf_files = ['example_config.ini']

    for conf_file in all_conf_files:
        configuration = library.get_config_from_ini(conf_file)

        cached_files = None
        if configuration.cache:
            print("Caching enabled. Preparing graphs...")
            # 1. Identify all required centralities
            all_centralities = set()
            for pair in configuration.centralities:
                all_centralities.add(pair[0])
                all_centralities.add(pair[1])
            all_centralities = list(all_centralities)

            # 2. Get existing files
            existing_files = library.get_cached_files(configuration)
            print(f"Found {len(existing_files)} existing files in cache.")

            # 3. Process graphs (Load/Generate -> Compute -> Save)
            # We need 'runs' graphs.
            # If runs > existing, we generate new ones.
            # If runs <= existing, we use the first 'runs' files (or all? usually runs).

            # We launch parallel processing to ensure they are ready.
            # Function: library.process_graph(index, config, cached_files, all_centralities)
            # We need to map indices 0..runs-1

            # NOTE: library.process_graph returns the filename.
            # We want to collect these filenames.

            process_args = [(i, configuration, existing_files, all_centralities) for i in range(configuration.runs)]

            with multiprocessing.Pool() as pool:
                cached_files = pool.starmap(library.process_graph, process_args)

            print(f"Prepared {len(cached_files)} graphs.")

        for centrality_pair in configuration.centralities:
            run_comparing_sim(configuration.runs, centrality_pair[0], centrality_pair[1], configuration, note=configuration.note, parallel=True, cached_files=cached_files)
            print(f"Completed simulation for {centrality_pair[0]} vs. {centrality_pair[1]}, configuration: {conf_file}")


if __name__ == "__main__":
    main()
