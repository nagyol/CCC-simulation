import typing
from functools import partial

import numpy as np

import library


def get_normalized_total_orderings(baseline: typing.Union[typing.List, typing.Dict], other: typing.Union[typing.List, typing.Dict]) -> typing.List:
    # Convert inputs to numpy arrays for speed
    n = len(baseline)

    # Check if inputs are dicts (mapping node_id -> score) or lists (index -> score)
    # The original code assumed keys 0..N-1 existed.
    if isinstance(baseline, dict):
        # Extract values in order of keys 0..N-1
        # This assumes keys are contiguous integers starting from 0, which matches the graph generation logic
        # (convert_node_labels_to_integers is used in library.py)
        baseline_arr = np.array([baseline[i] for i in range(n)])
    else:
        baseline_arr = np.array(baseline)

    if isinstance(other, dict):
        other_arr = np.array([other[i] for i in range(n)])
    else:
        other_arr = np.array(other)

    # Generate random tie-breakers
    random_arr = np.random.uniform(size=n)

    # Create indices
    indices = np.arange(n)

    # Sort for baseline: primary key baseline (desc), secondary other (desc), tertiary random (desc)
    # lexsort sorts by last key first. So we pass keys in reverse order of importance.
    # We want descending sort, so we negate the values.
    # keys: (-baseline, -other, -random)
    # lexsort order: -random, -other, -baseline

    keys_baseline = (-random_arr, -other_arr, -baseline_arr)
    # Note: lexsort is stable.
    normalized_baseline = indices[np.lexsort(keys_baseline)]

    # Sort for other: primary key other (desc), secondary baseline (desc), tertiary random (desc)
    keys_other = (-random_arr, -baseline_arr, -other_arr)
    normalized_other = indices[np.lexsort(keys_other)]

    return [normalized_baseline.tolist(), normalized_other.tolist()]


def compare_centrality(baseline: typing.List, ranking_other: typing.List) -> typing.Dict:
    norm_baseline, norm_other = get_normalized_total_orderings(baseline, ranking_other)

    n = len(baseline)
    # Convert to arrays for efficient processing
    nb = np.array(norm_baseline)
    no = np.array(norm_other)

    # seen_baseline[x] is True if node x has appeared in norm_baseline[:current_step]
    seen_baseline = np.zeros(n, dtype=bool)
    # seen_other[x] is True if node x has appeared in norm_other[:current_step]
    seen_other = np.zeros(n, dtype=bool)

    overlap_count = 0
    overlap_list = []

    # Pre-allocate output list for performance if N is large?
    # But python lists are dynamic. Appending is amortized O(1).
    # Since we need to return a list, we can just append.

    # It might be faster to do this loop in pure python if we can avoid overhead,
    # or use numba. But just avoiding set intersections is the main win.

    for i in range(n):
        u = nb[i] # node at rank i in baseline
        v = no[i] # node at rank i in other

        # Add u to seen_baseline
        seen_baseline[u] = True
        # If u was already seen in other, it means we found a match (u is in both top-sets)
        if seen_other[u]:
            overlap_count += 1

        # Add v to seen_other
        seen_other[v] = True
        # If v was already seen in baseline, match found
        if seen_baseline[v]:
            overlap_count += 1

        # If u == v, we incremented twice (once for u in seen_other? No).
        # if u == v:
        #   seen_baseline[u] = True
        #   if seen_other[u]: count++ (it's not true yet because we just set seen_other[v] below)

        # Let's trace u==v case carefully.
        # u == v.
        # seen_baseline[u] = True.
        # if seen_other[u]: (Assume False initially). No increment.
        # seen_other[u] = True.
        # if seen_baseline[u]: (True). Increment.
        # Count increases by 1. Correct.

        # Case u != v.
        # u added to baseline. If u in other, ++.
        # v added to other. If v in baseline, ++.
        # If u was already in other (from previous steps), we count it.
        # If v was already in baseline (from previous steps), we count it.
        # What if u and v are just appearing now?
        # u is new in baseline. v is new in other.
        # if u is in other (must be from prev steps).
        # if v is in baseline (must be from prev steps).
        # Correct.

        overlap_list.append(overlap_count)

    return {'overlap': overlap_list}


def plot_overlap(results: typing.List, baseline: typing.AnyStr, centrality: typing.AnyStr, zoom: bool = True,
                 note: typing.AnyStr = None) -> None:
    library.plot_general(results, baseline=baseline, centrality=centrality, header="", zoom=zoom,
                         parabola=True, rescale=True, note=note)
    return


def get_ranking(centrality: typing.AnyStr):
    lookup = {
        "betweenness": library.get_ranking_betweenness,
        "closeness": library.get_ranking_closeness,
        "harmonic": library.get_ranking_harmonic,
        "pagerank": library.get_ranking_pagerank,
        "degree": library.get_ranking_degree,
        "load": library.get_ranking_load,
        "katz": library.get_ranking_katz,
        "eigenvector": library.get_ranking_eigenvector,
        "indegree": library.get_ranking_indegree,
        "outdegree": library.get_ranking_outdegree
    }
    for damping_factor in range(0,100):
        lookup.update({f"pagerank-{damping_factor}": partial(library.get_ranking_pagerank, alpha=damping_factor/100)})

    return lookup[centrality]


def simulate_one_scenario(in_centrality1: typing.AnyStr, in_centrality2: typing.AnyStr, in_configuration: typing.NamedTuple):
    graph = in_configuration.generator()
    return compare_centrality(get_ranking(in_centrality1)(graph), get_ranking(in_centrality2)(graph))


def run_comparing_sim(runs: int, centrality1: typing.AnyStr, centrality2: typing.AnyStr, configuration: typing.NamedTuple,
                      note: typing.AnyStr = None, parallel: bool = True) -> None:

    print(f'Running: {centrality1} vs. {centrality2}, conffile: {configuration.name}')
    if not parallel:
        sim_results = []
        for i in range(runs):
            sim_results.append(simulate_one_scenario(centrality1, centrality2, configuration))
    else:
        sim_results = library.run_in_parallel(runs, partial(simulate_one_scenario, centrality1, centrality2, configuration))
    results = {'overlap': [res['overlap'] for res in sim_results]}
    plot_overlap(results['overlap'], baseline=centrality1, centrality=centrality2, zoom=False, note=note)


def main():

    all_conf_files = ['example_config.ini']
    for conf_file in all_conf_files:
        configuration = library.get_config_from_ini(conf_file)
        for centrality_pair in configuration.centralities:
            run_comparing_sim(configuration.runs, centrality_pair[0], centrality_pair[1], configuration, note=configuration.note, parallel=True)
            print(f"Completed simulation for {centrality_pair[0]} vs. {centrality_pair[1]}, configuration: {conf_file}")


if __name__ == "__main__":
    main()
